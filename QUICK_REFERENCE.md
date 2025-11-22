# Quick Reference

## Common Commands

### Run Tests
```bash
pytest tests/ -v                                    # All tests (63)
pytest tests/test_saturation_analysis.py -v        # Analysis tests
```

### Analyze Baseline Results
```bash
# Single table experiments
python -m icecap.saturation_analysis \
    -i experiments -p "exp2_1_*" -o plots/exp2_1

# Multi-table experiments
python -m icecap.saturation_analysis \
    -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables

# All baseline experiments
python -m icecap.saturation_analysis \
    -i experiments -p "exp2_*" -o plots/baseline
```

### View Results
```bash
# Experiment index (parameters + statistics)
cat plots/exp2_1/experiment_index.csv

# View plots
xdg-open plots/exp2_1/latency_vs_throughput.png      # Linux
open plots/exp2_1/latency_vs_throughput.png          # macOS
explorer.exe plots/exp2_1/                           # WSL
```

### Run Experiments (if needed)
```bash
# Full baseline (~9.5 hours)
nohup ./scripts/run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Monitor progress
./scripts/monitor_experiments.sh --watch 5

# Quick test
./scripts/run_baseline_experiments.sh --quick --seeds 1
```

## File Locations

| What | Where |
|------|-------|
| Documentation | `docs/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| Configs | `experiment_configs/` |
| Results | `experiments/` |
| Plots | `plots/` |

## Documentation

| Document | Purpose |
|----------|---------|
| [docs/README.md](docs/README.md) | Documentation index |
| [QUICKSTART.md](QUICKSTART.md) | Get started |
| [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md) | Analyze results |
| [docs/RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) | Run experiments |
| [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md) | Research plan |

## Project Status

- ✅ Simulator implemented and tested
- ✅ Baseline experiments complete
- ✅ Analysis system tested (63 tests passing)
- ✅ Documentation organized
- ⏳ Ready for analysis and visualization
- ⏳ Phase 3 experiments pending
