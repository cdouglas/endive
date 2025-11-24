# Icecap: Iceberg Catalog Simulator

A discrete-event simulator for characterizing throughput limits and commit latency in Apache Iceberg's optimistic concurrency control under varying workloads.

## What This Simulates

Apache Iceberg uses optimistic concurrency control (OCC) with compare-and-swap (CAS) operations. When multiple writers commit simultaneously to the same table:

1. Failed CAS triggers conflict resolution
2. Transaction reads manifest lists for missed snapshots
3. For **false conflicts** (version changed, no data overlap): reads metadata only (~1ms)
4. For **real conflicts** (overlapping changes): reads and rewrites manifest files (~400ms+)
5. Transaction retries commit with exponential backoff

This simulator explores: **When does commit throughput saturate? What causes latency to explode?**

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

### Outstanding Questions (Ready to Answer)

**Questions 1b & 2b** (Real conflicts) - Simulator ready, experiments not yet run:
- How do real conflicts shift saturation point?
- Cost difference: false (~1ms) vs real (~400ms)?
- How do real conflicts interact with table count?

## Quick Start

```bash
# Setup
python3 -m venv .
source bin/activate
pip install -r requirements.txt
pip install -e .

# Run baseline experiments (24 hours with 8 cores)
./scripts/run_baseline_experiments.sh --seeds 3

# Analyze results
python -m icecap.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1
python -m icecap.saturation_analysis -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables

# View results
open plots/exp2_1/latency_vs_throughput.png
cat plots/exp2_1/experiment_index.csv
```

**Quick test (2 minutes)**:
```bash
./scripts/run_baseline_experiments.sh --quick --seeds 1
```

## Docker

Run experiments in containers:

```bash
# Build and run baseline experiments
docker-compose up

# Run analysis only (after experiments complete)
docker-compose run --rm analyze

# Run conformance tests
docker-compose run --rm test
```

See [docs/DOCKER.md](docs/DOCKER.md) for details.

## Documentation

### Getting Started
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Installation, first simulation, analysis
- **[docs/DOCKER.md](docs/DOCKER.md)** - Container-based execution

### Understanding Results
- **[docs/BASELINE_RESULTS.md](docs/BASELINE_RESULTS.md)** - Detailed findings from Exp 2.1 & 2.2
- **[docs/OVERHEAD_ANALYSIS.md](docs/OVERHEAD_ANALYSIS.md)** - Commit protocol overhead breakdown

### Research & Design
- **[docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)** - Research questions and methodology
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Simulator design and invariants
- **[docs/SNAPSHOT_VERSIONING.md](docs/SNAPSHOT_VERSIONING.md)** - Version tracking details

### Configuration & Experiments
- **[experiment_configs/README.md](experiment_configs/README.md)** - Experiment templates
- **[docs/WARMUP_PERIOD.md](docs/WARMUP_PERIOD.md)** - Steady-state measurement methodology

### Complete Index
- **[docs/README.md](docs/README.md)** - Full documentation index

## Project Structure

```
icecap/
├── icecap/                     # Core simulator
│   ├── main.py                 # Simulation engine (OCC, CAS, conflict resolution)
│   ├── capstats.py             # Statistics collection
│   ├── saturation_analysis.py  # Saturation curve generation
│   └── warmup_validation.py    # Steady-state validation
├── experiment_configs/         # Experiment templates
│   ├── exp2_1_single_table_false_conflicts.toml
│   ├── exp2_2_multi_table_false_conflicts.toml
│   ├── exp3_1_single_table_real_conflicts.toml   # Ready to run
│   └── exp3_3_multi_table_real_conflicts.toml    # Ready to run
├── scripts/                    # Automation
│   ├── run_baseline_experiments.sh  # Parallel experiment runner
│   └── plot_distributions.py        # Distribution visualization
├── tests/                      # Test suite (63 tests)
├── docs/                       # Documentation
├── experiments/                # Results (189 baseline runs completed)
└── plots/                      # Analysis outputs
```

## Current Status

### ✅ Completed
- Core simulator with real/false conflict distinction
- Baseline experiments (Exp 2.1 & 2.2): 189 simulations across 5 seeds
- Saturation analysis with overhead measurement
- Distribution conformance tests
- Comprehensive documentation
- Docker support

### 📊 Available Results
- **Single-table saturation**: 60 c/s peak, 26% overhead
- **Multi-table scaling**: Sub-linear with coordination cost paradox
- **Latency curves**: P50/P95/P99 vs throughput with success rates
- **Overhead analysis**: Commit protocol cost scaling with table count

### 🔬 Ready to Run
- **Exp 3.1**: Single-table real conflicts (Question 1b)
- **Exp 3.2**: Manifest count distribution variance
- **Exp 3.3**: Multi-table real conflicts (Question 2b)

## Key Configuration Parameters

```toml
[catalog]
num_tables = 10        # Number of tables
num_groups = 10        # Conflict granularity (=num_tables for table-level)

[transaction]
retry = 10             # Maximum retries
runtime.mean = 180000  # 3 minutes (realistic Iceberg transactions)
inter_arrival.scale = 500.0  # Offered load (~2 txn/sec)

# Conflict types
real_conflict_probability = 0.0  # 0.0=false only, 1.0=real only

# For real conflicts
conflicting_manifests.distribution = "exponential"
conflicting_manifests.mean = 3.0

[storage]
max_parallel = 4       # Parallel manifest operations

# Infinitely fast catalog (baseline)
T_CAS.mean = 1
T_METADATA_ROOT.read.mean = 1

# Realistic S3 storage
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_FILE.read.mean = 50
```

## Testing

```bash
pytest tests/ -v                    # All 63 tests
pytest tests/ --cov=icecap          # With coverage
```

**Test coverage**: Core simulation, conflict resolution, experiment organization, analysis pipeline, warmup calculation

## Performance

- **Simulation speed**: ~1000× real-time (1 hour simulated in ~3.6 seconds)
- **Parallel execution**: ~8× speedup with 8 cores
- **Baseline runtime**: 24 hours with 8 cores (vs 8 days sequential)

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
- **Real conflicts will shift** saturation point significantly lower (experiments pending)

## Research Questions

This simulator was designed to answer:

1. **When does a single table saturate?** ✅ ~60 c/s with false conflicts
   - 1a. False conflicts only ✅ Answered (Exp 2.1)
   - 1b. With real conflicts ⏳ Ready to run (Exp 3.1)

2. **When do multi-table transactions saturate?** ✅ Sub-linear scaling
   - 2a. False conflicts only ✅ Answered (Exp 2.2)
   - 2b. With real conflicts ⏳ Ready to run (Exp 3.3)

See [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md) for complete methodology.

## Citation

(Add citation information here when published)

## References

- **Apache Iceberg**: https://iceberg.apache.org/
- **SimPy Framework**: https://simpy.readthedocs.io/
- **Optimistic Concurrency Control**: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
