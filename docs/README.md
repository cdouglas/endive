# Documentation Index

Complete documentation for the Iceberg Catalog Simulator.

## Quick Navigation

### 🚀 Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Installation, first simulation, analysis workflow
- **[DOCKER.md](DOCKER.md)** - Container-based execution with docker-compose

### 📊 Understanding Results
- **[BASELINE_RESULTS.md](BASELINE_RESULTS.md)** - Comprehensive findings from Exp 2.1 & 2.2
- **[OVERHEAD_ANALYSIS.md](OVERHEAD_ANALYSIS.md)** - Commit protocol overhead deep dive

### 🔬 Research & Methodology
- **[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)** - Research questions, current status, and methodology
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Simulator design, invariants, and implementation details
- **[SNAPSHOT_VERSIONING.md](SNAPSHOT_VERSIONING.md)** - Version tracking mechanics

### 📝 Additional Guides
- **[WARMUP_PERIOD.md](WARMUP_PERIOD.md)** - Steady-state measurement methodology
- **[ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)** - Analysis pipeline details
- **[RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)** - Parallel execution guide
- **[ORGANIZATION_SUMMARY.md](ORGANIZATION_SUMMARY.md)** - Experiment organization
- **[APPENDIX_SIMULATOR_DETAILS.md](APPENDIX_SIMULATOR_DETAILS.md)** - Technical appendix for blog posts
- **[CONSOLIDATED_FORMAT.md](CONSOLIDATED_FORMAT.md)** - Consolidated results format

## Documentation by Task

### I want to...

**...get started with the simulator**
→ Start with [QUICKSTART.md](QUICKSTART.md)

**...run experiments in Docker**
→ Follow [DOCKER.md](DOCKER.md)

**...run baseline experiments**
→ Follow [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md) or [QUICKSTART.md](QUICKSTART.md) "Run Baseline Experiments" section

**...analyze experiment results**
→ Follow [QUICKSTART.md](QUICKSTART.md) "Analyze Results" section or [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)

**...understand the key findings**
→ Read [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for comprehensive analysis results

**...understand overhead scaling**
→ Read [OVERHEAD_ANALYSIS.md](OVERHEAD_ANALYSIS.md)

**...create custom experiments**
→ See [../experiment_configs/README.md](../experiment_configs/README.md)

**...understand simulator design**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md)

**...understand version tracking**
→ Read [SNAPSHOT_VERSIONING.md](SNAPSHOT_VERSIONING.md)

**...understand the research plan**
→ Read [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)

**...validate warmup period**
→ Read [WARMUP_PERIOD.md](WARMUP_PERIOD.md)

**...troubleshoot issues**
→ Check [QUICKSTART.md](QUICKSTART.md) "Troubleshooting" section

## Documentation Structure

```
docs/
├── README.md (this file)           # Documentation index
│
├── Getting Started
│   ├── QUICKSTART.md               # Installation & first steps
│   └── DOCKER.md                   # Container-based execution
│
├── Results & Analysis
│   ├── BASELINE_RESULTS.md         # Exp 2.1 & 2.2 findings
│   ├── OVERHEAD_ANALYSIS.md        # Commit overhead deep dive
│   ├── ANALYSIS_GUIDE.md           # Analysis workflow
│   └── CONSOLIDATED_FORMAT.md      # Consolidated results format
│
├── Research & Design
│   ├── ANALYSIS_PLAN.md            # Research questions & status
│   ├── ARCHITECTURE.md             # Simulator design & invariants
│   └── SNAPSHOT_VERSIONING.md      # Version tracking mechanics
│
└── Advanced Topics
    ├── WARMUP_PERIOD.md            # Steady-state methodology
    ├── RUNNING_EXPERIMENTS.md      # Parallel execution
    ├── ORGANIZATION_SUMMARY.md     # Experiment organization
    └── APPENDIX_SIMULATOR_DETAILS.md  # Technical appendix for blog posts
```

## Project Structure

```
icecap/
├── README.md                       # Project overview with key findings
│
├── docs/                           # Documentation (you are here)
│   ├── README.md                   # This index
│   ├── QUICKSTART.md               # Getting started guide
│   ├── DOCKER.md                   # Docker execution
│   ├── BASELINE_RESULTS.md         # Experimental results
│   ├── OVERHEAD_ANALYSIS.md        # Overhead analysis
│   ├── ANALYSIS_PLAN.md            # Research plan
│   ├── ARCHITECTURE.md             # Simulator design
│   ├── SNAPSHOT_VERSIONING.md      # Version tracking
│   └── ... (other guides)
│
├── icecap/                         # Core simulator
│   ├── main.py                     # Simulation engine
│   ├── capstats.py                 # Statistics collection
│   ├── saturation_analysis.py      # Saturation analysis
│   └── warmup_validation.py        # Steady-state validation
│
├── experiment_configs/             # Experiment templates
│   ├── README.md                   # Configuration guide
│   ├── exp2_1_*.toml               # Single-table experiments
│   ├── exp2_2_*.toml               # Multi-table experiments
│   └── exp3_*.toml                 # Real conflict experiments (ready)
│
├── scripts/                        # Automation
│   ├── run_baseline_experiments.sh # Parallel experiment runner
│   └── plot_distributions.py       # Distribution visualization
│
├── tests/                          # Test suite (63 tests)
├── experiments/                    # Results (created at runtime)
└── plots/                          # Analysis outputs
```

## Quick Reference

### Common Commands

```bash
# Run experiments
./scripts/run_baseline_experiments.sh --seeds 3         # Full baseline (24h with 8 cores)
./scripts/run_baseline_experiments.sh --quick --seeds 1 # Quick test (2min)

# Analyze
python -m icecap.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1
python -m icecap.saturation_analysis -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables

# Validate
python -m icecap.warmup_validation experiments/exp2_1_*/12345

# Test
pytest tests/ -v
```

### File Locations

| What | Where |
|------|-------|
| Documentation | `docs/` (you are here) |
| Experiment configs | `../experiment_configs/` |
| Scripts | `../scripts/` |
| Results | `../experiments/` |
| Plots | `../plots/` |
| Tests | `../tests/` |
| Core simulator | `../icecap/main.py` |

### Key Metrics

| Metric | Where to Find |
|--------|---------------|
| Success rate | `plots/*/experiment_index.csv` |
| Throughput | `plots/*/experiment_index.csv` (commits/sec) |
| Latency percentiles | `plots/*/latency_vs_throughput.{png,md}` |
| Overhead % | `plots/*/overhead_vs_throughput.md` |
| Retries | `plots/*/experiment_index.csv` (mean_retries) |

## Current Status

### ✅ Completed
- Core simulator with real/false conflict distinction
- Baseline experiments (Exp 2.1 & 2.2): 189 simulations across 5 seeds
- Saturation analysis with overhead measurement
- Distribution conformance tests
- Comprehensive documentation
- Docker support

### 📊 Key Findings Available
- **Single-table saturation**: ~60 commits/sec peak, 26% overhead
- **Multi-table scaling**: Sub-linear with coordination cost paradox
- **Latency curves**: P50/P95/P99 vs throughput with success rates
- **Overhead analysis**: Commit protocol cost scaling with table count

### 🔬 Ready to Run
- **Exp 3.1**: Single-table real conflicts (Question 1b)
- **Exp 3.2**: Manifest count distribution variance
- **Exp 3.3**: Multi-table real conflicts (Question 2b)

## Key Concepts

### Experiment Organization

**Directory structure**: `experiments/$label-$hash/$seed/results.parquet`

- **$label**: Experiment name (e.g., "exp2_1_single_table_false")
- **$hash**: 8-char hash of config + code (excludes seed/label)
- **$seed**: Random seed for this run

**Benefits**: Same config + code → same hash for easy organization and reproducibility

### Saturation Point

Throughput at which success rate drops below 50%.

**Key metrics:**
- Saturation throughput (max sustainable commits/sec)
- P95 latency at saturation
- Mean retries per transaction

### False vs Real Conflicts

- **False conflict**: Version changed, no data overlap
  - Cost: ~1ms (metadata read only with fast catalog)
  - No manifest file operations

- **Real conflict**: Overlapping data changes
  - Cost: ~400ms+ (read/write N manifest files)
  - N sampled from configurable distribution

### Warmup Period

Transient exclusion for steady-state measurement:
- Duration: `max(5min, min(3 × mean_runtime, 15min))`
- Baseline: 9 minutes for 3-minute mean transactions
- Purpose: Allow queue depths to stabilize

## Common Workflows

### Workflow 1: First Time Setup

1. Read [QUICKSTART.md](QUICKSTART.md) installation section
2. Install: `pip install -r requirements.txt && pip install -e .`
3. Test: `pytest tests/ -v`
4. Run single simulation: `echo "Y" | python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml`

### Workflow 2: Running Baseline Experiments

1. Review [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) for context
2. Start: `./scripts/run_baseline_experiments.sh --seeds 3`
3. Monitor: `./scripts/monitor_experiments.sh --watch 5`
4. Wait: ~24 hours with 8 cores
5. Analyze: See Workflow 3

### Workflow 3: Analyzing Results

1. Generate saturation curves:
   ```bash
   python -m icecap.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1
   python -m icecap.saturation_analysis -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables
   ```

2. View outputs:
   - `plots/*/experiment_index.csv` - Summary statistics
   - `plots/*/latency_vs_throughput.{png,md}` - Main results
   - `plots/*/overhead_vs_throughput.md` - Overhead analysis

3. Review findings:
   - Read generated markdown tables
   - Compare [BASELINE_RESULTS.md](BASELINE_RESULTS.md)

### Workflow 4: Docker Execution

1. Read [DOCKER.md](DOCKER.md)
2. Build and run: `docker-compose up`
3. Analysis: `docker-compose run --rm analyze`
4. Tests: `docker-compose run --rm test`

## Getting Help

1. **Installation issues**: [QUICKSTART.md](QUICKSTART.md) troubleshooting
2. **Analysis questions**: [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)
3. **Docker issues**: [DOCKER.md](DOCKER.md) troubleshooting
4. **Understanding results**: [BASELINE_RESULTS.md](BASELINE_RESULTS.md)
5. **Simulator details**: [ARCHITECTURE.md](ARCHITECTURE.md)

## Contributing

Before making changes:
1. Run `pytest tests/ -v` to ensure tests pass
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for design invariants
3. Update documentation for new features
4. Add tests for new functionality

## External References

- **Apache Iceberg**: https://iceberg.apache.org/
- **SimPy Documentation**: https://simpy.readthedocs.io/
- **Optimistic Concurrency Control**: https://en.wikipedia.org/wiki/Optimistic_concurrency_control
