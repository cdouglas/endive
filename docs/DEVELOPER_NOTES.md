# Developer Notes & Quick Reference

This document provides quick reference information to help navigate and modify the codebase efficiently, reducing the need for extensive file reading in future sessions.

## Quick Architecture Overview

```
endive/
├── main.py              # Simulation entry point (1088 lines)
│   ├── SIM_SEED: Global for random seed (from config or auto-generated)
│   ├── prepare_experiment_output(): Creates experiments/$label-$hash/$seed/
│   └── main(): Runs simulation, uses seed from config.simulation.seed
│
├── saturation_analysis.py  # Analysis tool (1551 lines)
│   ├── CONFIG: Global config (loaded from analysis.toml or defaults)
│   ├── build_experiment_index(): Scans experiments/, filters by min_seeds (default: 3)
│   ├── load_and_aggregate_results(): Loads individual files, applies warmup/cooldown
│   ├── load_and_aggregate_results_consolidated(): Uses consolidated.parquet with predicate pushdown
│   └── Key filtering: Line 642-647 filters experiments with < min_seeds
│
└── capstats.py          # Statistics collection

scripts/
├── prepare_missing_experiments.sh   # Creates bundle with seed-specific configs
├── run_experiments.sh              # Auto-generated, runs batch in parallel
├── merge_experiment_results.sh     # Merges results back to experiments/
└── consolidate.py                             # Creates consolidated.parquet

tests/
└── test_saturation_analysis.py     # 36 tests for analysis pipeline
    └── CLI tests need min_seeds=1 in config to avoid filtering
```

## Key Implementation Details

### Seed Handling (main.py)
```python
# Line 220: Seed is read from config
SIM_SEED = config["simulation"].get("seed")

# Line 1078-1085: Seed initialization
if SIM_SEED is not None:
    seed = SIM_SEED
else:
    seed = np.random.randint(0, 2**32 - 1)
np.random.seed(seed)

# No --seed CLI argument! Must be in config file.
```

### Analysis Filtering (saturation_analysis.py)
```python
# Line 642-647: Experiments filtered by seed count
num_seeds = len(exp_info['seeds'])
min_seeds = CONFIG.get('analysis', {}).get('min_seeds', 3)  # Default: 3
if num_seeds < min_seeds:
    print(f" skipped (only {num_seeds} seeds, need {min_seeds})")
    continue
```

### Consolidated File Path Resolution (saturation_analysis.py)
```python
# Lines 626-646: Must resolve relative to base_dir
consolidated_path = os.path.join(base_dir, 'consolidated.parquet')
if os.path.exists(consolidated_path):
    df = load_and_aggregate_results_consolidated(exp_info, consolidated_path)
else:
    df = load_and_aggregate_results(exp_info)
```

## Common Issues & Solutions

### Issue: Tests failing with empty index
**Symptom**: `assert 0 == 1` (expected experiments, got 0)
**Causes**:
1. Test experiments have < 3 seeds → Set `min_seeds=1` in test config
2. Consolidated file exists but doesn't contain test data → Fixed in commit 0dd5d39
3. Insufficient data after warmup/cooldown → Increase transaction count to 300

**Solution**: Test configs need:
```toml
[analysis]
min_seeds = 1
```

### Issue: Docker execution fails with "unrecognized arguments: --seed"
**Symptom**: `main.py: error: unrecognized arguments: --seed 3083532295`
**Cause**: endive.main doesn't accept --seed argument

**Solution**: Set seed in config file:
```toml
[simulation]
seed = 3083532295
```

### Issue: Experiment batch preparation
**Pattern**: For remote execution, create separate config files with seeds pre-set:
```bash
# Wrong: Try to pass --seed at runtime
python -m endive.main cfg.toml --seed 12345  # FAILS

# Right: Pre-create configs with seeds
sed "s/^# seed = .*/seed = 12345/" cfg.toml > seed_12345.toml
python -m endive.main seed_12345.toml
```

## Common grep Patterns

```bash
# Find where seed is used
grep -n "SIM_SEED\|\.get.*seed" endive/main.py

# Find filtering logic
grep -n "min_seeds\|< min_seeds" endive/saturation_analysis.py

# Find config loading
grep -n "CONFIG\s*=" endive/saturation_analysis.py
```

## File Size Reference
```
endive/main.py:                  1088 lines (~35KB)
endive/saturation_analysis.py:   1551 lines (~55KB)
tests/test_saturation_analysis.py: 1260 lines (~45KB)
scripts/prepare_missing_experiments.sh: 352 lines (~12KB)
```

## Key Commits (Recent)

- `0dd5d39`: Fixed consolidated file path resolution bug (tests now pass)
- `a4facb3`: Pre-set seeds in config files during preparation
- `8fb86d2`: Added rerun_missing_experiments.sh
- `3e4654e`: Added remote execution workflow (prepare/merge scripts)
- `0e3d277`: Added docker-compose batch profile

## Token-Saving Strategies

### When debugging test failures:
1. First check this doc for common issues
2. Use targeted grep instead of reading full files:
   ```bash
   grep -A5 -B5 "specific_pattern" file.py
   ```
3. Only read relevant test functions, not entire test file

### When modifying analysis code:
1. Check architecture overview above for function locations
2. Use Edit tool with specific old_string/new_string
3. Avoid reading entire saturation_analysis.py (1551 lines)

### When working with experiments:
1. Use find/ls to check structure before reading configs
2. Sample one config file, assume others are similar
3. Use head -30 instead of reading full configs

### General:
1. Prefer Grep with output_mode="content" and context flags (-A, -B, -C)
2. Use Read with offset/limit for large files
3. Document patterns immediately when found
4. Avoid re-reading files within same session

## Docker Quick Reference

```bash
# Interactive container
docker run -it \
    -v "$(pwd)/experiment_batch:/app/experiment_batch" \
    cdouglas/endive-sim:latest bash

# Using docker-compose
BATCH_PARALLEL=8 docker-compose --profile batch up

# Available profiles: batch, analysis, test, single
```

## Experiment Structure

```
experiments/
├── exp_name-hash/          # Experiment directory
│   ├── cfg.toml           # Configuration (copied from experiment_configs/)
│   └── seed/              # One directory per seed
│       └── results.parquet
```

## Analysis Configuration Defaults

```toml
[analysis]
min_seeds = 3                    # Minimum seeds per experiment
k_min_cycles = 5                 # For warmup calculation
min_warmup_ms = 300000           # 5 minutes
max_warmup_ms = 900000           # 15 minutes
use_consolidated = true          # Use consolidated.parquet if available

[paths]
consolidated_file = "experiments/consolidated.parquet"
```

## Testing Notes

- Total tests: 109 (all passing as of commit a4facb3)
- Test categories: 36 saturation_analysis, 11 regression, 28 simulator, etc.
- CLI tests require min_seeds=1 to avoid filtering
- Tests create temporary experiments in /tmp

## Performance Notes

- Simulation speed: ~1000× real-time (1 hour sim in ~3.6 seconds)
- Storage: consolidated.parquet provides efficient compression
- Consolidation time: ~22 minutes for all experiments (incremental script)
- Each experiment takes ~1 hour to run

## Future Optimization Ideas

1. Create function signature reference for main.py and saturation_analysis.py
2. Add quick index of all grep patterns used in debugging
3. Document common sed/awk patterns for config manipulation
4. Create visualization of data flow through analysis pipeline
