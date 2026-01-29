# Running Baseline Experiments

Quick reference guide for running Phase 2 baseline experiments.

## Quick Start

### 1. Run experiments in background

```bash
# Full baseline experiments (Phase 2) with 3 seeds per configuration
nohup ./scripts/run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Save the process ID
echo $! > experiments.pid
```

This will run:
- **Experiment 2.1**: Single table, 9 load levels, 3 seeds = 27 simulations
- **Experiment 2.2**: Multi-table, 6 table counts × 9 load levels × 3 seeds = 162 simulations
- **Total**: 189 simulations (~9.5 hours estimated)

### 2. Monitor progress

```bash
# Watch with auto-refresh every 5 seconds
./scripts/monitor_experiments.sh --watch 5

# One-time status check
./scripts/monitor_experiments.sh --summary
```

### 3. Check if still running

```bash
# Check process status
ps aux | grep run_baseline_experiments.sh

# Or use saved PID
if ps -p $(cat experiments.pid) > /dev/null; then
    echo "Experiments still running"
else
    echo "Experiments complete or stopped"
fi

# View latest log output
tail -f experiment_logs/baseline_experiments_*.log
```

## Common Scenarios

### Quick Test Run

Before running full experiments, do a quick test:

```bash
# Quick test mode: fewer configs, 10-second simulations
./run_baseline_experiments.sh --quick --seeds 1

# Expected: ~2-3 minutes for full test
# Verifies: setup, config generation, result output
```

### Run Only Experiment 2.1

```bash
# Single table saturation only (faster)
nohup ./run_baseline_experiments.sh --exp2.1 --seeds 3 > exp2_1.log 2>&1 &
```

### Run Only Experiment 2.2

```bash
# Multi-table saturation only
nohup ./run_baseline_experiments.sh --exp2.2 --seeds 3 > exp2_2.log 2>&1 &
```

### Higher Confidence Results

```bash
# Run with 10 seeds per configuration for more stable statistics
# Warning: Takes ~3x longer!
nohup ./run_baseline_experiments.sh --seeds 10 > baseline_10seeds.log 2>&1 &
```

### Dry Run

```bash
# See what would be executed without running
./run_baseline_experiments.sh --dry-run --seeds 3
```

## Understanding the Output

### Log Files

```bash
experiment_logs/
└── baseline_experiments_YYYYMMDD_HHMMSS.log
```

Contains:
- Configuration summary
- Progress updates
- Success/failure status for each run
- Final summary with counts

### Experiment Results

```bash
experiments/
├── exp2_1_single_table_false-HASH/
│   ├── cfg.toml                  # Shared configuration
│   ├── SEED1/results.parquet     # First run
│   ├── SEED2/results.parquet     # Second run
│   └── SEED3/results.parquet     # Third run
└── exp2_2_multi_table_false-HASH/
    ├── cfg.toml
    └── ...
```

Multiple hash values will appear because each load level and table count creates a different configuration (different hash).

### Monitoring Output

```
========================================
EXPERIMENT PROGRESS MONITOR
========================================
Time: 2025-11-21 16:45:30

Log file: experiment_logs/baseline_experiments_20251121_164000.log

Overall Progress: 45 / 189 (23%)
Remaining: 144 simulations
[===========---------------------------------------] 23%

Results:
  ✓ Successful: 44
  ✗ Failed: 1

Current Activity:
  Load: inter_arrival.scale = 100ms
  [45/189]  Seed 2/3
  ✓ Success
```

## Time Estimates

Based on 100-second simulations with ~3 minutes total per run:

| Configuration | Simulations | Est. Time |
|--------------|-------------|-----------|
| Full baseline (3 seeds) | 189 | ~9.5 hours |
| Exp 2.1 only (3 seeds) | 27 | ~1.5 hours |
| Exp 2.2 only (3 seeds) | 162 | ~8 hours |
| Quick test (1 seed) | ~6 | ~2 minutes |
| Full baseline (10 seeds) | 630 | ~31 hours |

**Note:** Actual times vary based on:
- Load level (higher load = more retries = longer runtime)
- System performance
- Other running processes

## Troubleshooting

### Experiments Failed to Start

```bash
# Check for errors in latest log
tail -50 experiment_logs/baseline_experiments_*.log

# Common issues:
# 1. Virtual environment not activated
source bin/activate

# 2. Module not installed
pip install -e .

# 3. Config files not found
ls experiment_configs/*.toml
```

### High Failure Rate

```bash
# Check specific error messages
grep "✗ Failed" -A 5 experiment_logs/baseline_experiments_*.log

# Common causes:
# - Disk space full (check: df -h)
# - Permission issues (check: ls -l experiments/)
# - Config syntax errors (validate: python -m endive.main <config> --dry-run)
```

### Kill Running Experiments

```bash
# If you need to stop experiments
kill $(cat experiments.pid)

# Or find and kill manually
ps aux | grep run_baseline_experiments.sh
kill <PID>

# The experiments can be resumed - already completed runs are saved
```

### Resume After Interruption

The experiment script doesn't automatically resume, but you can:

```bash
# Check what's already completed
ls -R experiments/exp2_*

# Manually count completed runs per configuration
# Then restart with remaining configs
# (Future enhancement: add resume capability)
```

## Next Steps After Completion

### 1. Verify Results

```bash
# Check experiment directories were created
ls -lh experiments/

# Count result files
find experiments/ -name "results.parquet" | wc -l

# Expected: 189 for full baseline with 3 seeds
```

### 2. Run Analysis

See `ANALYSIS_GUIDE.md` (to be created) or:

```bash
# Generate plots for Experiment 2.1
python -m endive.analysis all \
    -i experiments/exp2_1_* \
    -o plots/exp2_1

# Generate plots for Experiment 2.2
python -m endive.analysis all \
    -i experiments/exp2_2_* \
    -o plots/exp2_2
```

### 3. Proceed to Phase 3

After baseline experiments complete, run Phase 3 (real conflicts):

```bash
# To be created: run_real_conflict_experiments.sh
```

## Command Reference

```bash
# Run full baseline experiments
./run_baseline_experiments.sh --seeds 3

# Run in background
nohup ./run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &

# Monitor progress
./monitor_experiments.sh --watch 5

# Quick test
./run_baseline_experiments.sh --quick --seeds 1

# Dry run
./run_baseline_experiments.sh --dry-run

# Specific experiment only
./run_baseline_experiments.sh --exp2.1 --seeds 3
./run_baseline_experiments.sh --exp2.2 --seeds 3

# Help
./run_baseline_experiments.sh --help
./monitor_experiments.sh --help
```

## Tips

1. **Test first**: Always run `--quick --seeds 1` before full experiments
2. **Save PID**: Keep track of background process for easy monitoring
3. **Watch logs**: Use `tail -f` on log files to see real-time progress
4. **Disk space**: Check available space before starting (experiments can use several GB)
5. **Patience**: Full baseline experiments take ~9.5 hours - let them run overnight
6. **Multiple seeds**: 3 seeds provide reasonable confidence; 5-10 seeds for publication-quality results

## File Locations

```
.
├── run_baseline_experiments.sh       # Main experiment runner
├── monitor_experiments.sh            # Progress monitor
├── experiment_configs/               # Configuration templates
│   ├── exp2_1_single_table_false_conflicts.toml
│   ├── exp2_2_multi_table_false_conflicts.toml
│   └── README.md
├── experiment_logs/                  # Execution logs
│   └── baseline_experiments_*.log
└── experiments/                      # Results (created during run)
    ├── exp2_1_single_table_false-*/
    └── exp2_2_multi_table_false-*/
```
