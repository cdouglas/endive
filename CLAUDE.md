# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Endive is a discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC). It models catalog contention, conflict resolution, and commit latency under varying workloads. Its purpose is to evaluate changes to the commit protocol and table format.

## Task Tracking (beads)

**Use `bd` (beads) for ALL task tracking. Do NOT use TaskCreate/TaskUpdate/TaskList.**

- Before starting work: `bd ready` to find available tasks, or `bd create` for new ones
- When starting: `bd update <id> --status=in_progress`
- When done: `bd close <id> --reason="what was done"`
- Session end: `bd sync` before committing

**Session close checklist (in order):**
1. `bd close` all completed issues
2. `bd sync`
3. `git add` + `git commit`
4. `bd sync`
5. `git push`

Leaving beads issues in `in_progress` blocks git push hooks. Always close before finishing.

## Development Commands

### Setup
```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Running Tests
```bash
# All tests (136 tests, ~3 minutes)
pytest tests/ -v

# Specific test suites
pytest tests/test_simulator.py -v                      # Core simulator (28 tests)
pytest tests/test_saturation_analysis*.py -v           # Analysis pipeline (36 tests)
pytest tests/test_numerical_accuracy.py -v             # Numerical validation (8 tests)
pytest tests/test_statistical_rigor.py -v              # Distribution conformance (6 tests)

# Run single test
pytest tests/test_simulator.py::test_deterministic_seed -v

# With coverage
pytest tests/ --cov=endive --cov-report=html

# Fast subset (core tests only, ~30 seconds)
pytest tests/test_simulator.py tests/test_conflict_resolution.py -v
```

### Running Simulations
```bash
# Single experiment (1 hour simulation, ~3.6 seconds wall-clock)
python -m endive.main experiment_configs/exp2_1_single_table_false_conflicts.toml --yes

# With specific seed
python -m endive.main my_config.toml --seed 42 --yes

# Quick test mode (1 minute duration, fewer params)
python scripts/run_all_experiments.py --quick --seeds 1

# Run specific experiment groups
python scripts/run_all_experiments.py --groups baseline,heatmap --seeds 3

# Background with logging
nohup python scripts/run_all_experiments.py --seeds 3 > experiments.log 2>&1 &
```

### Analysis
```bash
# Regenerate all plots (reads [plots] sections from experiment configs)
python scripts/regenerate_plots.py

# Dry run (show what would be generated)
python scripts/regenerate_plots.py --dry-run

# Single experiment config
python scripts/regenerate_plots.py --config experiment_configs/exp1_fa_baseline.toml

# Only configs matching pattern
python scripts/regenerate_plots.py --pattern "exp3*"

# Direct saturation_analysis CLI (for ad-hoc analysis)
python -m endive.saturation_analysis -i experiments -p "exp1_fa_baseline-*" -o plots/exp1_fa_baseline
python -m endive.saturation_analysis -i experiments -p "exp2_mix_heatmap-*" -o plots/exp2_mix_heatmap --group-by fast_append_ratio

# Consolidate results (reduces storage by ~60% with compression)
python scripts/consolidate.py
```

### Docker
```bash
# Build image
docker build -t cdouglas/endive-sim:latest .

# Run experiments in container
docker run -d \
    -e DOCKER_CONTAINER=1 \
    -e OMP_NUM_THREADS=1 \
    -v $(pwd)/experiments:/app/experiments \
    -v $(pwd)/plots:/app/plots \
    cdouglas/endive-sim:latest \
    bash -c "python scripts/run_all_experiments.py --groups baseline,metadata --seeds 5 --parallel 8"

# See docs/DOCKER.md for full details
```

## Architecture Overview

### Core Components

**`endive/main.py` (1088 lines)** - Simulation engine
- `Catalog` class: Versioned state with compare-and-swap operations
- `Txn` class: Transaction lifecycle (capture snapshot → execute → commit → retry)
- `ConflictResolver` class: Distinguishes false vs real conflicts, handles manifest operations
- `txn_gen()`: Transaction generator with configurable inter-arrival distributions
- `calculate_backoff_time()`: Exponential backoff with jitter

**`endive/saturation_analysis.py` (1700+ lines)** - Analysis pipeline
- `build_experiment_index()`: Scans experiments/, extracts parameters from cfg.toml files
- `load_and_aggregate_results()`: Loads individual per-seed parquet files
- `load_and_aggregate_results_consolidated()`: Efficient loading from consolidated.parquet with predicate pushdown
- `plot_*()` functions: Generate saturation curves, overhead analysis, time-series plots
- `parse_filter_expression()` / `apply_filters()`: Filter experiments by parameter values

**`endive/capstats.py`** - Statistics collection during simulation

**`scripts/run_all_experiments.py`** - Unified experiment runner
- Supports experiment groups (baseline, heatmap, catalog)
- Parameter sweeps with deterministic seed generation (nonce-based)
- Progress tracking, resume capability, status checking
- Parallel execution with configurable concurrency

**`scripts/regenerate_plots.py`** - Unified plot regeneration
- Reads `[plots]` sections from experiment configs, dispatches to plotting functions
- Merges `plotting.toml` defaults with per-graph overrides
- Supports `--dry-run`, `--pattern`, `--config` for targeted regeneration

### Critical Design Patterns

#### 1. Snapshot Versioning (main.py:359-393, 527)
```python
# Transaction captures catalog version at creation
txn.v_catalog_seq = catalog.seq  # Line 527

# On commit, checks current version
success = (catalog.seq == txn.v_catalog_seq)  # CAS check

# On failure, reads EXACTLY n manifest lists for n missed snapshots
n_missed = catalog.seq - txn.v_catalog_seq
# Must read one manifest list per intermediate snapshot
```

**Critical invariant**: `catalog.seq` advances by exactly 1 on each successful commit. Never skip versions.

#### 2. Conflict Types (main.py:1600-1777)
```python
# False conflict: same table modified, but different partitions (no data overlap)
# In rewrite mode: Read ML + Write new ML (~100ms) - must merge ML pointers
# In ML+ mode: No ML operations needed (~10ms) - tentative entry still valid
yield from resolve_false_conflict(...)

# Real conflict: overlapping data changes (same partition)
# Cost: Read ML + Read/Write N manifest files + Write ML (~400ms+)
yield from resolve_real_conflict(...)
```

**Key distinction**:
- False conflicts: No manifest FILE operations, but ML operations depend on mode
- Real conflicts: Require expensive manifest file I/O in both modes
- ML+ advantage: Saves ~100ms per false conflict by avoiding ML read/write on retry

#### 3. Table Grouping (main.py:52-131)
```python
# Partition tables into G groups
# Transactions NEVER span group boundaries
# G=1: Catalog-level conflicts (all transactions conflict)
# G=T: Table-level conflicts (only same-table writes conflict)
```

**Critical invariant**: Transaction isolation is per-group. Cross-group transactions would violate the model.

#### 4. Experiment Organization
```
experiments/
├── exp2_1_single_table_false-a3f7b2/    # Label + hash of parameters
│   ├── cfg.toml                          # Configuration snapshot
│   ├── 42/results.parquet                # Seed 42 results
│   ├── 43/results.parquet                # Seed 43 results
│   └── 44/results.parquet                # Seed 44 results
└── consolidated.parquet                  # All experiments (v2.0+)
```

**Hash computation**: `compute_experiment_hash()` creates deterministic hash from parameters (excludes seed, output_path). Same parameters → same hash → same directory.

#### 5. Seeds and Determinism
**IMPORTANT**: Seeds must be in config file, NOT passed as CLI argument to endive.main:
```bash
# WRONG: endive.main doesn't accept --seed
python -m endive.main config.toml --seed 42  # FAILS

# RIGHT: Set seed in config file
[simulation]
seed = 42
```

For batch experiments, use `run_all_experiments.py` which handles config variants and deterministic seed generation via nonce.

#### 6. Consolidated Format (v2.0+)
- Single parquet file with all experiments: `experiments/consolidated.parquet`
- Uses predicate pushdown for efficient filtering (memory efficiency, not speed)
- Falls back to individual files if consolidated doesn't exist
- Regenerate with: `python scripts/consolidate.py`

### Analysis Pipeline Flow

1. **Index Building** (`build_experiment_index()`):
   - Scans `experiments/` for pattern matches (e.g., "exp2_1_*")
   - Reads `cfg.toml` from each experiment directory
   - Extracts parameters: `inter_arrival_scale`, `num_tables`, `real_conflict_probability`, `t_cas_mean`, etc.
   - Filters by `min_seeds` (default: 3)

2. **Data Loading** (`load_and_aggregate_results_consolidated()` or `load_and_aggregate_results()`):
   - Loads transaction-level data from parquet files
   - Applies warmup/cooldown filtering based on steady-state calculation
   - Aggregates across seeds

3. **Statistics** (`compute_aggregate_statistics()`):
   - Throughput, success rate, latency percentiles (P50/P95/P99)
   - Retry statistics, overhead calculation
   - Standard deviations across seeds

4. **Plotting** (`plot_*()` functions):
   - Latency vs throughput curves with error bands
   - Success rate degradation
   - Overhead analysis
   - Time-series commit rate plots

### Parameter Filtering

For multi-dimensional parameter sweeps (e.g., exp3.3: num_tables × real_conflict_probability × inter_arrival):

```bash
# Filter to single num_tables value, group by real_conflict_probability
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp3_3_*" \
    -o plots/exp3_3_t10 \
    --group-by real_conflict_probability \
    --filter "num_tables==10"

# Multiple filters (AND logic) - use SEPARATE --filter arguments
--filter "num_tables>=5" --filter "real_conflict_probability<=0.5"

# WRONG: Do NOT use && operator (not supported)
--filter "num_tables==10 && real_conflict_probability==0.3"  # FAILS
```

**Supported operators**: `==`, `!=`, `<`, `<=`, `>`, `>=`
**Filter logic**: Multiple `--filter` arguments are AND'd together

## Common Workflows

### Adding a New Experiment

1. Create config file in `experiment_configs/`:
```toml
[simulation]
duration_ms = 3600000
output_path = "results.parquet"

[experiment]
label = "exp_my_test"  # Will create experiments/exp_my_test-<hash>/

[catalog]
num_tables = 1
num_groups = 1

[transaction]
retry = 10
runtime.mean = 180000
runtime.sigma = 1.5
inter_arrival.scale = 100.0
real_conflict_probability = 0.0
```

2. Add to `EXPERIMENT_GROUPS` in `scripts/run_all_experiments.py` if doing parameter sweeps

3. Add `[plots]` section to the config file declaring which graphs to generate

### Debugging Test Failures

**Common issues**:

1. **Empty experiment index** (`assert expected_count == 0`):
   - Test experiments have < 3 seeds → Set `min_seeds=1` in test config
   - Consolidated file exists but doesn't contain test data → Use fresh test environment

2. **Seed-related failures**:
   - Remember: Seeds must be in config file, not CLI argument
   - Check `SIM_SEED` global in main.py:220

3. **Numerical accuracy issues**:
   - Simulator is accurate to machine epsilon (1e-10)
   - Check for floating-point accumulation in long simulations
   - See tests/test_numerical_accuracy.py for validation patterns

### Modifying Analysis Code

**Key locations**:
- Parameter extraction: `saturation_analysis.py:289-324` (`extract_key_parameters()`)
- Warmup calculation: `saturation_analysis.py:269-278` (`compute_warmup_duration()`)
- Filtering: `saturation_analysis.py:1473-1588`
- Plotting: `saturation_analysis.py:692-1135`

**Pattern**: Use targeted edits instead of reading entire 1700-line file. Use grep with context:
```bash
grep -A 10 -B 5 "def extract_key_parameters" endive/saturation_analysis.py
```

### Working with Results

```bash
# View experiment metadata
cat experiments/exp2_1_single_table_false-a3f7b2/cfg.toml

# Quick statistics
python -c "import pandas as pd; df = pd.read_parquet('experiments/exp2_1_*/42/results.parquet'); print(df['status'].value_counts())"

# Check consolidation
python -c "import pyarrow.parquet as pq; meta = pq.read_metadata('experiments/consolidated.parquet'); print(f'{meta.num_rows:,} rows, {meta.num_row_groups} row groups')"
```

## Key Documentation

- **README.md** - Concise getting started guide (installation, usage, analysis, testing)
- **docs/APPENDIX_SIMULATOR_DETAILS.md** - Technical appendix for blog posts (distributions, parameters, formulas)
- **docs/ARCHITECTURE.md** - Detailed design, invariants, code locations
- **docs/QUICKSTART.md** - Installation and first simulation walkthrough
- **docs/ANALYSIS_GUIDE.md** - How to generate plots and interpret results
- **docs/CONSOLIDATED_FORMAT.md** - Consolidated parquet format details
- **docs/DEVELOPER_NOTES.md** - Common issues, token-saving strategies, quick reference
- **docs/DOCKER.md** - Container-based execution with EXP_ARGS
- **docs/BASELINE_RESULTS.md** - Key findings from baseline experiments
- **experiment_configs/README.md** - Experiment descriptions and parameter sweeps

## Important Constraints

1. **Never modify catalog.seq directly** - Only increment via successful CAS
2. **Transaction-group boundaries** - Transactions never span groups
3. **Warmup/cooldown periods** - Always filter transient data before analysis
4. **Seed determinism** - Same seed + same config = bitwise identical results
5. **Manifest list reads** - Must read exactly n lists for n missed snapshots
6. **Config file seeds** - Seeds go in TOML, not CLI arguments to endive.main
7. **Hash stability** - Changing parameter extraction breaks experiment directory matching

## Performance Notes

- **Simulation speed**: ~1000× real-time (1 hour sim in ~3.6 seconds)
- **Parallel execution**: Near-linear speedup with CPU core count
- **Baseline runtime**: ~24 hours with 8 cores for full experiment suite
- **Memory usage**: Analysis loads ~200MB per experiment with consolidated format
- **Test execution**: 136 tests in ~3 minutes

## Current Experiment Coverage

**Experiment Groups (run via `--groups`):**
- **baseline**: 100% FastAppend with instant catalog (`exp1_fa_baseline`)
- **heatmap**: FA/VO operation mix 2D sweep (`exp2_mix_heatmap`)
- **catalog**: Catalog latency impact on FA and mixed workloads (`exp3a_catalog_fa`, `exp3b_catalog_mix`)
- **tables**: Multi-table single-file catalog contention (`exp4a_tables_fa`, `exp4b_tables_mix`)

**Plotting Approach:**
- Each experiment config declares `[plots]` section with positive list of graphs
- `plotting.toml` provides function-mapped defaults; experiment configs override per-graph
- Automatic generation via `python scripts/regenerate_plots.py`
- Direct CLI via `python -m endive.saturation_analysis` for ad-hoc analysis
