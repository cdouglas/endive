# Icecap Documentation Index

Complete documentation for the Icecap simulator and saturation analysis framework.

## Quick Start

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [QUICKSTART.md](../QUICKSTART.md) | Get started quickly | First time setup |
| [Running Experiments](RUNNING_EXPERIMENTS.md) | Run baseline experiments | Before starting experiments |
| [Analysis Guide](ANALYSIS_GUIDE.md) | Analyze results | After experiments complete |

## Core Documentation

### Getting Started

1. **[QUICKSTART.md](../QUICKSTART.md)**
   - Installation and setup
   - First simulation
   - Basic configuration
   - Understanding results

2. **[README.md](../README.md)**
   - Complete feature overview
   - Architecture details
   - Configuration reference
   - Testing guide

### Running Experiments

3. **[Running Experiments Guide](RUNNING_EXPERIMENTS.md)**
   - Baseline experiment scripts
   - Parameter sweeps
   - Monitoring progress
   - Troubleshooting

4. **[Experiment Configs](../experiment_configs/README.md)**
   - Configuration templates
   - Parameter descriptions
   - Experiment designs

### Analysis and Visualization

5. **[Analysis Guide](ANALYSIS_GUIDE.md)**
   - Saturation analysis workflow
   - Generating plots
   - Interpreting results
   - Custom analysis

6. **[Analysis Summary](ANALYSIS_SUMMARY.md)**
   - System overview
   - Problem solved
   - Output structure
   - Commands reference

### Research and Planning

7. **[ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md)**
   - Research questions
   - Experiment methodology
   - Implementation phases
   - Deliverables

## Documentation by Task

### I want to...

**...get started with the simulator**
→ Start with [QUICKSTART.md](../QUICKSTART.md)

**...understand simulator features**
→ Read [README.md](../README.md) sections on:
- Configuration
- Table Grouping
- Conflict Resolution
- Storage Latencies

**...run baseline experiments**
→ Follow [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md)

**...create custom experiments**
→ See [experiment_configs/README.md](../experiment_configs/README.md)

**...analyze experiment results**
→ Follow [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)

**...generate latency vs throughput plots**
→ Use saturation analysis (see [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md#quick-start))

**...understand the research plan**
→ Read [ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md)

**...troubleshoot issues**
→ Check troubleshooting sections in:
- [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md#troubleshooting)
- [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md#troubleshooting)
- [QUICKSTART.md](../QUICKSTART.md#troubleshooting)

## File Organization

```
.
├── README.md                          # Main project overview
├── QUICKSTART.md                      # Quick start guide
├── ANALYSIS_PLAN.md                   # Research methodology
│
├── docs/                              # Documentation
│   ├── README.md                      # This file (index)
│   ├── RUNNING_EXPERIMENTS.md         # Experiment execution
│   ├── ANALYSIS_GUIDE.md              # Analysis workflow
│   └── ANALYSIS_SUMMARY.md            # System overview
│
├── experiment_configs/                # Experiment templates
│   ├── README.md                      # Config guide
│   ├── exp2_1_*.toml                  # Single table experiments
│   ├── exp2_2_*.toml                  # Multi-table experiments
│   └── exp3_*.toml                    # Real conflict experiments
│
├── icecap/                            # Simulator code
│   ├── main.py                        # Core simulator
│   ├── capstats.py                    # Statistics
│   ├── analysis.py                    # Legacy analysis
│   └── saturation_analysis.py         # New analysis module
│
├── tests/                             # Test suite
│   ├── test_simulator.py              # Core tests
│   ├── test_conflict_types.py         # Conflict resolution
│   ├── test_experiment_structure.py   # Organization tests
│   └── test_saturation_analysis.py    # Analysis tests
│
└── scripts/                           # Automation scripts
    ├── run_baseline_experiments.sh    # Experiment runner
    └── monitor_experiments.sh         # Progress monitor
```

## Common Workflows

### Workflow 1: First Time User

1. Read [QUICKSTART.md](../QUICKSTART.md)
2. Run a single simulation: `python -m icecap.main cfg.toml`
3. Explore configuration: edit `cfg.toml`
4. Run tests: `pytest tests/ -v`

### Workflow 2: Running Baseline Experiments

1. Review [experiment_configs/README.md](../experiment_configs/README.md)
2. Start experiments: `./run_baseline_experiments.sh --seeds 3`
3. Monitor: `./monitor_experiments.sh --watch 5`
4. Wait ~9.5 hours
5. Analyze: Follow [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)

### Workflow 3: Analyzing Results

1. Build experiment index:
   ```bash
   python -m icecap.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1
   ```

2. View generated files:
   - `plots/exp2_1/experiment_index.csv` - Parameter index
   - `plots/exp2_1/latency_vs_throughput.png` - Main plot
   - `plots/exp2_1/success_rate_vs_load.png` - Secondary plot

3. Extract findings:
   ```bash
   cat plots/exp2_1/experiment_index.csv
   ```

4. Custom analysis:
   ```python
   import pandas as pd
   df = pd.read_csv('plots/exp2_1/experiment_index.csv')
   # Your analysis here
   ```

### Workflow 4: Creating Custom Experiments

1. Copy a template:
   ```bash
   cp experiment_configs/exp2_1_*.toml my_experiment.toml
   ```

2. Edit parameters:
   - Modify `inter_arrival.scale`
   - Adjust `real_conflict_probability`
   - Change `num_tables`

3. Add experiment label:
   ```toml
   [experiment]
   label = "my_custom_experiment"
   ```

4. Run:
   ```bash
   python -m icecap.main my_experiment.toml
   ```

## Key Concepts

### Experiment Organization

**Structure:** `experiments/$label-$hash/$seed/results.parquet`

- **$label**: Experiment name (e.g., "exp2_1_single_table_false")
- **$hash**: 8-character hash of config (excluding seed/label) + code
- **$seed**: Random seed used for this run

**Benefit:** Same config + code → same hash, making it easy to:
- Find related runs
- Run multiple seeds
- Track code changes

### Saturation Point

The throughput at which success rate drops below 50%.

**Key metrics:**
- **Saturation throughput**: Max sustainable commits/sec
- **P95 latency at saturation**: 95th percentile commit time
- **Retry rate**: Average retries per transaction

### False vs Real Conflicts

- **False conflict**: Version changed, no data overlap
  - Cost: ~100ms (metadata read only)
  - No manifest file operations

- **Real conflict**: Overlapping data changes
  - Cost: ~500ms+ (read/merge/write manifest files)
  - Number of files sampled from distribution

### Latency vs Throughput Trade-off

- **Low load**: High success, low latency, low throughput
- **Medium load**: Balanced success and throughput
- **High load**: Low success, high latency, saturated throughput

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-21 | Initial documentation organization |
| | | - Added saturation analysis module |
| | | - Experiment runner scripts |
| | | - Comprehensive guides |

## Contributing

When adding new features:
1. Update relevant documentation
2. Add tests in `tests/`
3. Update this index if adding new docs
4. Run full test suite: `pytest tests/ -v`

## Support

For issues or questions:
1. Check [troubleshooting sections](#i-want-to)
2. Review [ANALYSIS_PLAN.md](../ANALYSIS_PLAN.md) for methodology
3. Examine test files for usage examples
4. Check experiment logs in `experiment_logs/`
