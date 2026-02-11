# Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | Installation and first simulation |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Simulator design and invariants |
| [model.md](model.md) | Model simplifications vs reality |
| [errata.md](errata.md) | Technical debt and gaps |

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
| [APPEND_SIMULATION_DESIGN.md](APPEND_SIMULATION_DESIGN.md) | ML+ append mode design |
| [SNAPSHOT_VERSIONING.md](SNAPSHOT_VERSIONING.md) | Version tracking mechanics |
| [WARMUP_PERIOD.md](WARMUP_PERIOD.md) | Steady-state measurement |
| [OVERHEAD_ANALYSIS.md](OVERHEAD_ANALYSIS.md) | Commit protocol overhead |
| [APPENDIX_SIMULATOR_DETAILS.md](APPENDIX_SIMULATOR_DETAILS.md) | Technical appendix for publications |

## Results

| Document | Description |
|----------|-------------|
| [BASELINE_RESULTS.md](BASELINE_RESULTS.md) | Findings from baseline experiments |

## Quick Reference

### Run Simulation
```bash
python -m endive.main experiment_configs/exp8_0_baseline_s3x.toml --yes
```

### Generate Plots
```bash
python -m endive.saturation_analysis -i experiments -p "exp8_*" -o plots/exp8 --group-by label
```

### Run Tests
```bash
pytest tests/ -v
```

## File Locations

| What | Where |
|------|-------|
| Experiment configs | `../experiment_configs/` |
| Results | `../experiments/` |
| Plots | `../plots/` |
| Core simulator | `../endive/main.py` |
| Analysis | `../endive/saturation_analysis.py` |
