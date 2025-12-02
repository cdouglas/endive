# Icecap: Iceberg Catalog Simulator

A discrete-event simulator for characterizing throughput limits and commit latency in Apache Iceberg's optimistic concurrency control under varying workloads.

## What This Simulates

Apache Iceberg uses optimistic concurrency control (OCC) with compare-and-swap (CAS) operations. When multiple writers commit simultaneously to the same table:

1. Failed CAS triggers conflict resolution
2. Transaction reads manifest lists for missed snapshots
3. For **false conflicts** (version changed, no data overlap): reads metadata only (~1ms)
4. For **real conflicts** (overlapping changes): reads and rewrites manifest files (~400ms+)
5. Transaction retries commit (optionally with exponential backoff)

This simulator explores: **When does commit throughput saturate? What causes latency to explode? Does exponential backoff help?**

## Key Findings

### Single-Table Saturation (Question 1a ✅)

**Peak throughput: ~60 commits/sec** with infinitely fast catalog (1ms CAS)

- **Efficient operating point**: 50 c/s @ 76% success rate, P95 latency = 131s
- **Saturation threshold**: 55-60 c/s (50% success rate)
- **At saturation**: 26% overhead (1/4 of time spent retrying commits)
- **Bottleneck**: Contention, not catalog speed

### Multi-Table Scaling (Question 2a ✅)

**Throughput scales sub-linearly** with table count

| Tables | Throughput | Scaling Efficiency | Overhead @ Saturation |
|--------|------------|-------------------|----------------------|
| 1 | 62 c/s | baseline | 26% |
| 2 | 94 c/s | 76% | 39% |
| 5 | 111 c/s | 32% | 46% |
| 10 | 121 c/s | 19% | 50% |
| 20 | 131 c/s | 11% | 54% |
| 50 | 144 c/s | 5% | **59%** |

**Key insight**: 50× more tables → only 2.3× throughput. Sweet spot: 10-20 tables.

**The Latency Paradox**: More tables increase throughput but WORSEN tail latency
- Why? Multi-table coordination cost dominates at high load
- At 50 tables: Commit protocol takes MORE time than transaction execution (59% overhead)

## Quick Start

### Installation

```bash
# Setup virtual environment
python3 -m venv .
source bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running Experiments

#### Single Experiment
```bash
# Run a single 1-hour simulation
python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml

# Run with specific seed
python -m icecap.main my_config.toml --seed 42

# Skip confirmation prompt
python -m icecap.main my_config.toml --yes
```

#### Batch Experiments
```bash
# Run baseline experiments (24 hours with 8 cores)
./scripts/run_baseline_experiments.sh --seeds 3

# Quick test (2 minutes, 1 seed)
./scripts/run_baseline_experiments.sh --quick --seeds 1

# Run specific experiment set
./scripts/run_baseline_experiments.sh --pattern "exp2_1_*" --seeds 5
```

### Consolidating Results

```bash
# Consolidate all experiments into single parquet file
python scripts/consolidate_all_experiments.py

# Result: experiments/consolidated.parquet
# Contains all experiment data with predicate pushdown support
# 22× faster analysis than reading individual files
```

The consolidated format (v2.0+) enables efficient filtering:
```python
import pyarrow.parquet as pq

# Load only specific experiments
table = pq.read_table(
    'experiments/consolidated.parquet',
    filters=[('experiment_label', '=', 'exp2_1_load_050')]
)
```

### Analyzing Results

#### Basic Analysis
```bash
# Analyze single experiment set
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Analyze with grouping (e.g., by number of tables)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables

# Filter by parameter values (useful for multi-dimensional sweeps)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp5_2_*" \
    -o plots/exp5_2_filtered \
    --group-by num_tables \
    --filter "t_cas_mean==50"

# Multiple filters (AND logic)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp5_2_*" \
    -o plots/filtered \
    --filter "t_cas_mean>=50" \
    --filter "num_tables<=5"

# Use custom configuration
python -m icecap.saturation_analysis --config analysis.toml
```

#### Configuration-Based Analysis

Create `analysis.toml`:
```toml
[analysis]
input_dir = "experiments"
output_dir = "plots/exp2_1"
pattern = "exp2_1_*"
k_min_cycles = 5          # Steady-state cycles
min_warmup_ms = 300000    # 5 minutes
max_warmup_ms = 900000    # 15 minutes

[plots.saturation]
enabled = true
threshold = 50.0          # Success rate threshold
tolerance = 5.0

[plots]
dpi = 300
[plots.figsize]
latency_vs_throughput = [12, 8]
```

Then run:
```bash
# Auto-discovers analysis.toml
python -m icecap.saturation_analysis

# Override specific options
python -m icecap.saturation_analysis \
    --config analysis.toml \
    -o custom_output \
    --group-by num_tables
```

### Generating Plots

The analysis script automatically generates:

**Saturation Curves**:
- `latency_vs_throughput.png` - P50/P95/P99 latency curves with success rates
- `success_rate_vs_throughput.png` - Success rate degradation
- `success_rate_vs_load.png` - Offered load vs actual throughput
- `overhead_vs_throughput.png` - Commit protocol overhead

**Distribution Analysis**:
```bash
# Visualize runtime and inter-arrival distributions
python scripts/plot_distributions.py experiments/exp2_1_load_050-<hash>/42/results.parquet

# Generates:
# - plots/distributions/transaction_runtime.png
# - plots/distributions/inter_arrival_times.png
```

**Experiment Index**:
- `experiment_index.csv` - Tabular summary of all experiments
- `experiment_index.md` - Markdown table for documentation

### Viewing Results

```bash
# View plots
open plots/exp2_1/latency_vs_throughput.png
open plots/exp2_1/experiment_index.csv

# View experiment metadata
cat experiments/exp2_1_load_050-<hash>/cfg.toml

# View raw data
python -c "import pandas as pd; print(pd.read_parquet('experiments/consolidated.parquet').head())"
```

## Configuration Examples

### Basic Experiment Configuration

```toml
[simulation]
duration_ms = 3600000      # 1 hour
seed = 42
output_path = "results.parquet"

[catalog]
num_tables = 10            # Number of tables
num_groups = 10            # Conflict granularity

[transaction]
retry = 10                 # Maximum retries
runtime.mean = 180000      # 3 minutes
runtime.sigma = 1.5        # Lognormal variance
inter_arrival.scale = 500.0  # ~2 txn/sec offered load

# Conflict configuration
real_conflict_probability = 0.0  # 0=false only, 1=real only
conflicting_manifests.distribution = "exponential"
conflicting_manifests.mean = 3.0

[storage]
max_parallel = 4           # Parallel manifest operations
min_latency = 5            # Minimum operation latency

# Fast catalog (baseline)
T_CAS.mean = 1
T_METADATA_ROOT.read.mean = 1
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_FILE.read.mean = 50
```

### Experiment with Label (Organized Output)

```toml
[simulation]
duration_ms = 3600000
output_path = "results.parquet"

[experiment]
label = "exp2_1_load_050"   # Creates experiments/exp2_1_load_050-<hash>/

[catalog]
num_tables = 1

# ... rest of config ...
```

Output structure:
```
experiments/
└── exp2_1_load_050-a3f7b2/
    ├── cfg.toml              # Configuration snapshot
    ├── 42/
    │   └── results.parquet   # Seed 42 results
    ├── 43/
    │   └── results.parquet   # Seed 43 results
    └── 44/
        └── results.parquet   # Seed 44 results
```

## Testing

### Test Suite Overview

The simulator includes a comprehensive test suite with **136 tests** covering:

**Core Simulator (109 tests)**:
- Determinism and reproducibility
- Conflict resolution and retry logic
- Snapshot versioning and CAS operations
- Table grouping and transaction isolation
- Distribution conformance (runtime, inter-arrival)
- Experiment organization and hashing
- Analysis pipeline accuracy

**Phase 1: Critical Gaps (21 tests)**:
- Reentrant execution (experiment recovery)
- Numerical accuracy (machine epsilon validation)
- Edge cases (extreme load, boundary conditions)

**Phase 2: Statistical Rigor (6 tests)**:
- Distribution conformance (K-S tests)
- Selection bias quantification
- Cross-experiment consistency

### Running Tests

```bash
# All tests (~3 minutes)
pytest tests/ -v

# Specific test suites
pytest tests/test_simulator.py -v                    # Core simulator
pytest tests/test_numerical_accuracy.py -v           # Numerical validation
pytest tests/test_statistical_rigor.py -v            # Statistical tests
pytest tests/test_saturation_analysis*.py -v         # Analysis pipeline

# With coverage
pytest tests/ --cov=icecap --cov-report=html

# Fast subset (core tests only)
pytest tests/test_simulator.py tests/test_conflict_resolution.py -v
```

### Test Validation Results

**✅ Simulator Correctness Validated**:
- Numerical accuracy to machine epsilon (1e-10)
- Perfect bitwise determinism with same seed
- No floating-point error accumulation
- Runtime distribution: lognormal (30% tolerance)
- Inter-arrival distribution: exponential (K-S test p > 0.01)
- Selection bias: quantified and understood

**✅ Zero Bugs Found**:
All test failures during development were test implementation issues, not simulator bugs. Previous simulation results remain valid.

See [docs/test_results_summary.md](docs/test_results_summary.md) for detailed test analysis.

## Docker

Run experiments in containers:

```bash
# Build and run baseline experiments
docker-compose up

# Run analysis only (after experiments complete)
docker-compose run --rm analyze

# Run conformance tests
docker-compose run --rm test

# Run batch experiments
docker-compose --profile batch up
```

See [docs/DOCKER.md](docs/DOCKER.md) for details.

## Project Structure

```
icecap/
├── icecap/                      # Core simulator
│   ├── main.py                  # Simulation engine (OCC, CAS, conflicts)
│   ├── capstats.py              # Statistics collection
│   ├── saturation_analysis.py   # Saturation curve generation
│   ├── warmup_validation.py     # Steady-state validation
│   └── test_utils.py            # Test utilities and builders
├── experiment_configs/          # Experiment templates
│   ├── exp2_1_single_table_false_conflicts.toml
│   ├── exp2_2_multi_table_false_conflicts.toml
│   ├── exp3_1_single_table_real_conflicts.toml
│   └── exp3_3_multi_table_real_conflicts.toml
├── scripts/                     # Automation
│   ├── run_baseline_experiments.sh      # Parallel experiment runner
│   ├── consolidate_all_experiments.py   # Result consolidation
│   ├── plot_distributions.py            # Distribution visualization
│   └── warmup_validation.py             # Warmup period analysis
├── tests/                       # Test suite (136 tests, 100% passing)
│   ├── test_simulator.py               # Core determinism
│   ├── test_conflict_resolution.py     # Conflict handling
│   ├── test_numerical_accuracy.py      # Calculation validation
│   ├── test_statistical_rigor.py       # Distribution tests
│   ├── test_edge_cases.py              # Boundary conditions
│   ├── test_reentrant_execution.py     # Experiment recovery
│   ├── test_saturation_analysis*.py    # Analysis pipeline
│   └── ...                              # Additional test modules
├── docs/                        # Documentation
│   ├── test_results_summary.md         # Test validation results
│   ├── test_coverage_assessment.md     # Coverage analysis
│   └── ...                              # Domain documentation
├── experiments/                 # Results (189 baseline runs)
│   └── consolidated.parquet     # Consolidated results (v2.0+)
└── plots/                       # Analysis outputs
    ├── exp2_1/                  # Single-table results
    └── exp2_2/                  # Multi-table results
```

## Performance

- **Simulation speed**: ~1000× real-time (1 hour simulated in ~3.6 seconds)
- **Parallel execution**: ~8× speedup with 8 cores
- **Baseline runtime**: 24 hours with 8 cores (vs 8 days sequential)
- **Analysis speed** (v2.0+): 22× faster with consolidated format
- **Test execution**: 136 tests in ~3 minutes

## Documentation

### Getting Started
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Installation, first simulation, analysis
- **[docs/DOCKER.md](docs/DOCKER.md)** - Container-based execution

### Understanding Results
- **[docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md)** - Detailed findings from Exp 2.1 & 2.2
- **[docs/OVERHEAD_ANALYSIS.md](docs/OVERHEAD_ANALYSIS.md)** - Commit protocol overhead breakdown
- **[docs/CONSOLIDATED_FORMAT.md](docs/CONSOLIDATED_FORMAT.md)** - Consolidated results format

### Testing & Validation
- **[docs/test_results_summary.md](docs/test_results_summary.md)** - Comprehensive test analysis
- **[docs/test_coverage_assessment.md](docs/test_coverage_assessment.md)** - Coverage gaps and plan
- **[docs/TEST_COVERAGE.md](docs/TEST_COVERAGE.md)** - Analysis pipeline tests

### Research & Design
- **[docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)** - Research questions and methodology
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Simulator design and invariants
- **[docs/SNAPSHOT_VERSIONING.md](docs/SNAPSHOT_VERSIONING.md)** - Version tracking details

### Configuration & Experiments
- **[experiment_configs/README.md](experiment_configs/README.md)** - Experiment templates
- **[docs/WARMUP_PERIOD.md](docs/WARMUP_PERIOD.md)** - Steady-state measurement

### Complete Index
- **[docs/README.md](docs/README.md)** - Full documentation index

## Practical Implications

### For Iceberg Users
- **Target load**: <30 commits/sec per table for good latency (P95 < 90s)
- **Acceptable range**: 30-50 c/s with degrading performance
- **Saturation zone**: 50-60 c/s (high retries, exponential latency growth)
- **Thrashing**: >60 c/s (>80% abort rate)

### For Catalog Designers
- **Faster catalogs help, but...** contention dominates above 50 c/s
- **Multi-table helps, but...** coordination overhead limits scaling
- **False conflicts cost less** but still expensive at scale (26% overhead @ saturation)
- **Real conflicts will shift** saturation point significantly lower

## Research Questions

This simulator was designed to answer:

1. **When does a single table saturate?** ✅ ~60 c/s with false conflicts
   - 1a. False conflicts only ✅ Answered (Exp 2.1)
   - 1b. With real conflicts ⏳ Ready to run (Exp 3.1)

2. **When do multi-table transactions saturate?** ✅ Sub-linear scaling
   - 2a. False conflicts only ✅ Answered (Exp 2.2)
   - 2b. With real conflicts ⏳ Ready to run (Exp 3.3)

See [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md) for complete methodology.

## Current Status

### ✅ Completed
- Core simulator with real/false conflict distinction
- Reentrant execution (experiment recovery)
- Exponential backoff implementation
- Baseline experiments (Exp 2.1 & 2.2): 189 simulations
- Saturation analysis with overhead measurement
- Consolidated results format (v2.0)
- Comprehensive test suite (136 tests, 100% passing)
- Statistical validation (K-S tests, bias quantification)
- Documentation and examples

### 📊 Available Results
- **Single-table saturation**: 60 c/s peak, 26% overhead
- **Multi-table scaling**: Sub-linear with coordination cost paradox
- **Latency curves**: P50/P95/P99 vs throughput with success rates
- **Overhead analysis**: Commit protocol cost scaling

### 🔬 Ready to Run
- **Exp 3.1**: Single-table real conflicts (Question 1b)
- **Exp 3.2**: Manifest count distribution variance
- **Exp 3.3**: Multi-table real conflicts (Question 2b)

## Citation

(Add citation information here when published)

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **SimPy Framework**: https://simpy.readthedocs.io/
- **Optimistic Concurrency Control**: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
