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

```

## Architecture Overview

See **SPEC.md** for the authoritative module layout, APIs, and invariants.

### Core Components

**`endive/simulation.py`** - SimPy runner. The only module that touches SimPy; bridges bare `float` latency generators to `env.timeout()` via `_drive_generator()`. Owns `Simulation`, `SimulationConfig`, `Statistics` (streaming parquet export).

**`endive/catalog.py`** - `Catalog` ABC plus `CASCatalog`, `AppendCatalog`, `InstantCatalog`. Exposes only `read()` and `commit()`. Manages per-table, per-partition version vectors.

**`endive/transaction.py`** - `Transaction` ABC plus `FastAppendTransaction` and `ValidatedOverwriteTransaction`. Owns the commit loop, per-attempt I/O cost model, write-overlap detection, and conflict-cost calculation.

**`endive/storage.py`** - `StorageProvider` ABC, `LognormalLatency`/`SizeBasedLatency`/`FixedLatency`, and concrete providers (S3, S3 Express, Azure, Azure Premium, GCS, instant).

**`endive/conflict_detector.py`** - `ProbabilisticConflictDetector` and `PartitionOverlapConflictDetector`.

**`endive/workload.py`** - `Workload`, `WorkloadConfig`, table/partition selectors (uniform, Zipf). Owns topology.

**`endive/config.py`** - `load_simulation_config()`; the only config entry point. Loads TOML, builds provider profiles, constructs all components.

**`endive/main.py`** - CLI entry point (~330 lines) and experiment directory management.

**`endive/saturation_analysis.py`** - Analysis pipeline: `build_experiment_index()`, `load_and_aggregate_results_consolidated()`, statistics, plotting functions.

**`scripts/run_all_experiments.py`** - Experiment runner. Supports experiment groups (baseline, heatmap, catalog, tables, zipf, providers, partition, inlined). Deterministic seed generation via nonce. Resume capability, status checking, parallel execution.

**`scripts/regenerate_plots.py`** - Reads `[plots]` sections from experiment configs, dispatches to plotting functions. Merges `plotting.toml` defaults with per-graph overrides. Supports `--dry-run`, `--pattern`, `--config`.

### Critical Design Patterns

#### 1. Generator-Based I/O
All latency-bearing operations yield bare `float` values (milliseconds). Only `Simulation._drive_generator()` converts these to SimPy timeouts. This separates I/O modeling from the simulation engine.

#### 2. Snapshot Versioning
Transactions observe catalog state only through immutable `CatalogSnapshot`s returned by `catalog.read()`. On commit, `catalog.commit(expected_seq, writes)` performs CAS against the expected seq. On failure, the transaction calls `catalog.read()` again and uses `compute_write_overlap()` to decide whether manifest I/O is needed.

**Critical invariant**: `catalog.seq` advances by exactly 1 on each successful commit. Never skip versions.

#### 3. Write Overlap Check
After a CAS failure, `Transaction.compute_write_overlap()` compares per-partition version vectors between the transaction's start snapshot and the current snapshot. Cross-table or disjoint-partition retries skip all manifest I/O — catalog read + re-CAS only. Same-table overlapping-partition retries pay type-specific conflict costs (FA: nothing extra; VO: historical ML reads for the I/O convoy).

#### 4. Conflict Types
- **No overlap** (cross-table or disjoint partitions): Free retry (catalog read + CAS only)
- **False conflict**: Same table + overlapping partitions, no data conflict — merge and retry with per-attempt I/O
- **Real conflict**: Same table + overlapping partitions with data conflict — may abort (operation-dependent)

#### 5. Experiment Organization
```
experiments/
├── exp1_fa_baseline-a3f7b2/         # Label + hash of parameters
│   ├── cfg.toml                     # Configuration snapshot
│   ├── version.txt                  # Git SHA
│   ├── 42/results.parquet           # Seed 42 results
│   └── 43/results.parquet           # Seed 43 results
└── consolidated.parquet             # All experiments merged
```

**Hash computation**: `compute_experiment_hash()` creates a deterministic hash from config parameters (excludes seed, output_path, experiment.label). Same parameters + same code → same hash → same directory.

#### 6. Seeds and Determinism
**IMPORTANT**: Seeds go in the config file, not as a CLI argument to `endive.main`:
```bash
# WRONG: endive.main doesn't accept --seed
python -m endive.main config.toml --seed 42  # FAILS

# RIGHT: Set seed in config file
[simulation]
seed = 42
```

For batch experiments, use `run_all_experiments.py` which handles config variants and deterministic seed generation via a nonce.

#### 7. Consolidated Format
- Single parquet file with all experiments: `experiments/consolidated.parquet`
- Uses predicate pushdown for efficient filtering (memory efficiency, not speed)
- Falls back to individual files if consolidated doesn't exist
- Regenerate with: `python scripts/consolidate.py`

### Analysis Pipeline Flow

1. **Index Building** (`build_experiment_index()`):
   - Scans `experiments/` for pattern matches (e.g., "exp1_*")
   - Reads `cfg.toml` from each experiment directory
   - Extracts parameters: `inter_arrival_scale`, `num_tables`, `real_conflict_probability`, `fast_append_ratio`, etc.
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

For multi-dimensional parameter sweeps (e.g., exp4b: num_tables × catalog_latency_ms × inter_arrival):

```bash
# Filter to a single num_tables value, group by catalog latency
python -m endive.saturation_analysis \
    -i experiments \
    -p "exp4b_*" \
    -o plots/exp4b_t10 \
    --group-by catalog_latency_ms \
    --filter "num_tables==10"

# Multiple filters (AND logic) — use SEPARATE --filter arguments
--filter "num_tables>=5" --filter "catalog_latency_ms<=50"

# WRONG: Do NOT use && operator (not supported)
--filter "num_tables==10 && catalog_latency_ms==50"  # FAILS
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
   - Test experiments have < 3 seeds → set `min_seeds=1` in the test config
   - Consolidated file exists but doesn't contain test data → use a fresh test environment

2. **Seed-related failures**:
   - Seeds must be in the TOML config, not a CLI argument to `endive.main`
   - `SimulationConfig.seed` is applied in `simulation.Simulation.run()`

### Modifying Analysis Code

Use grep with context instead of reading the full 4k-line `saturation_analysis.py`:
```bash
grep -A 10 -B 5 "def extract_key_parameters" endive/saturation_analysis.py
```

### Working with Results

```bash
# View experiment metadata
cat experiments/exp1_fa_baseline-*/cfg.toml

# Quick statistics on a single seed
python -c "import pandas as pd; df = pd.read_parquet('experiments/exp1_fa_baseline-*/42/results.parquet'); print(df['status'].value_counts())"

# Check the consolidated file
python -c "import pyarrow.parquet as pq; meta = pq.read_metadata('experiments/consolidated.parquet'); print(f'{meta.num_rows:,} rows, {meta.num_row_groups} row groups')"
```

## Key Documentation

- **SPEC.md** - Authoritative simulator specification (module APIs, invariants, config schema)
- **README.md** - Concise getting started guide (installation, usage, analysis, testing)
- **docs/model.md** - Simplifications relative to real Iceberg
- **docs/EXP5.md**, **docs/EXP6.md** - Partition-aware and inlined-metadata experiment designs
- **docs/CONSOLIDATED_FORMAT.md** - Consolidated parquet format details
- **docs/analysis/** - Provider latency verification, DES profiling, reference data
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
- **Test execution**: ~878 tests in ~3 minutes

## Current Experiment Coverage

**Experiment Groups (run via `--groups`):**
- **baseline**: 100% FastAppend single-table saturation (`exp1_fa_baseline`)
- **heatmap**: FA/VO operation mix 2D sweep (`exp2_mix_heatmap`)
- **catalog**: Catalog CAS latency × FA/mixed workloads (`exp3a_catalog_fa`, `exp3b_catalog_mix`)
- **tables**: Multi-table single-file catalog contention (`exp4a_tables_fa`, `exp4b_tables_mix`)
- **zipf**: Zipf table selection (`exp4a_zipf_tables_fa`, `exp4b_zipf_tables_mix`)
- **providers**: Real provider profiles × tables × workload (`exp4c_tables_providers`)
- **partition**: Partition-aware single table (`exp5[ab]_[zipf_]partition_{fa,mix}`)
- **inlined**: Inlined table metadata (`exp6[ab]_[zipf_]inlined_{fa,mix}`)

**Plotting Approach:**
- Each experiment config declares `[plots]` section with positive list of graphs
- `plotting.toml` provides function-mapped defaults; experiment configs override per-graph
- Automatic generation via `python scripts/regenerate_plots.py`
- Direct CLI via `python -m endive.saturation_analysis` for ad-hoc analysis


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
