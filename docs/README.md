# Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Installation and first simulation |
| [model.md](model.md) | Model simplifications vs reality |
| [APPENDIX_SIMULATOR_DETAILS.md](APPENDIX_SIMULATOR_DETAILS.md) | Technical appendix for publications |
| [SIMULATOR_REVIEW.md](SIMULATOR_REVIEW.md) | Fidelity analysis vs Apache Iceberg source |

## Guides

| Document | Description |
|----------|-------------|
| [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) | Generating and interpreting plots |
| [RUNNING_EXPERIMENTS.md](RUNNING_EXPERIMENTS.md) | Parallel experiment execution |
| [DOCKER.md](DOCKER.md) | Container-based execution |
| [CONSOLIDATED_FORMAT.md](CONSOLIDATED_FORMAT.md) | Results storage format |

## Technical Details

| Document | Description |
|----------|-------------|
| [SNAPSHOT_VERSIONING.md](SNAPSHOT_VERSIONING.md) | Version tracking mechanics |
| [WARMUP_PERIOD.md](WARMUP_PERIOD.md) | Steady-state measurement |
| [OVERHEAD_ANALYSIS.md](OVERHEAD_ANALYSIS.md) | Commit protocol overhead |
| [IO_CONVOY_ANALYSIS.md](IO_CONVOY_ANALYSIS.md) | Historical ML read I/O convoy |

## Results

| Document | Description |
|----------|-------------|
| [BASELINE_RESULTS.md](BASELINE_RESULTS.md) | Findings from baseline experiments |

## Quick Reference

### Run Simulation
```bash
python -m endive.main experiment_configs/exp1_fa_baseline.toml --yes
```

### Generate Plots
```bash
python scripts/regenerate_plots.py
```

### Run Tests
```bash
pytest tests/ -v
```

## File Locations

| What | Where |
|------|-------|
| Specification | `../SPEC.md` |
| Experiment configs | `../experiment_configs/` |
| Results | `../experiments/` |
| Plots | `../plots/` |
| Core simulator | `../endive/simulation.py` |
| Analysis | `../endive/saturation_analysis.py` |
