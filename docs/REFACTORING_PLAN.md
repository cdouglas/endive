# Refactoring Plan: endive/main.py → Modules

## Overview

Split `endive/main.py` (3263 lines) into logical modules while maintaining:
- **Exact determinism**: Same seed → identical results before/after
- **Test compatibility**: All 382 tests pass without modification
- **Import compatibility**: `from endive.main import X` continues to work

## Current Structure Analysis

| Section | Lines | Functions/Classes |
|---------|-------|-------------------|
| Git utilities | 20-50 | `get_git_sha`, `get_git_sha_short` |
| Global config | 50-145 | 50+ global variables |
| ContentionTracker | 147-215 | `ContentionTracker` class |
| Latency utilities | 424-610 | `lognormal_mu_from_median`, `parse_latency_config`, etc. |
| Table grouping | 610-690 | `partition_tables_into_groups` |
| Config hash | 692-745 | `compute_experiment_hash` |
| Config loading | 746-1070 | `configure_from_toml` |
| Config validation | 1070-1250 | `validate_config` |
| Latency generators | 1251-1560 | `get_*_latency`, `generate_*`, `calculate_backoff_time` |
| Print/confirm | 1559-1645 | `print_configuration`, `confirm_run` |
| ConflictResolver | 1645-2092 | `ConflictResolver` class |
| Catalog | 2092-2300 | `Catalog` class |
| Txn | 2301-2324 | `Txn` dataclass |
| LogEntry | 2324-2336 | `LogEntry` dataclass |
| AppendCatalog | 2336-2555 | `AppendCatalog` class |
| Transaction ops | 2555-2820 | `txn_ml_w`, `txn_commit`, etc. |
| Partition selection | 2819-2858 | `select_partitions` |
| Txn generation | 2858-2974 | `rand_tbl`, `txn_gen` |
| Simulation | 2974-3000 | `setup` |
| Experiment mgmt | 2988-3117 | `check_existing_experiment`, `prepare_experiment_output` |
| CLI | 3117-3263 | `cli` |

## Target Structure

```
endive/
├── __init__.py          # Public API re-exports
├── main.py              # CLI entry point (slim, ~150 lines)
├── config.py            # Configuration & globals (~400 lines)
├── latency.py           # Latency generation (~350 lines)
├── catalog.py           # Catalog classes (~500 lines)
├── conflict.py          # ConflictResolver (~450 lines)
├── transaction.py       # Txn + commit functions (~500 lines)
├── simulation.py        # Simulation runner (~200 lines)
├── experiment.py        # Experiment management (~150 lines)
└── utils.py             # Utilities (~100 lines)
```

## Module Responsibilities

### `config.py` (~400 lines)
- All global variable declarations
- `ContentionTracker` class
- `configure_from_toml()`
- `validate_config()`
- `parse_latency_config()`, `get_provider_latency_config()`, `apply_provider_defaults()`
- Provider configuration constants

### `latency.py` (~350 lines)
- `lognormal_mu_from_median()`, `convert_mean_stddev_to_lognormal()`
- `generate_latency()`, `generate_latency_lognormal()`, `generate_latency_from_config()`
- `generate_inter_arrival_time()`
- All `get_*_latency()` functions
- `sample_conflicting_manifests()`, `calculate_backoff_time()`

### `catalog.py` (~500 lines)
- `Catalog` class
- `AppendCatalog` class
- `LogEntry` dataclass

### `conflict.py` (~450 lines)
- `ConflictResolver` class (entire class)

### `transaction.py` (~500 lines)
- `Txn` dataclass
- `select_partitions()`
- `rand_tbl()`
- `txn_gen()`
- `txn_ml_w()`, `txn_ml_append()`
- `txn_commit()`, `txn_commit_append()`

### `simulation.py` (~200 lines)
- `setup()`
- Main simulation loop logic
- `print_configuration()`, `confirm_run()`

### `experiment.py` (~150 lines)
- `check_existing_experiment()`
- `prepare_experiment_output()`
- `compute_experiment_hash()`

### `utils.py` (~100 lines)
- `get_git_sha()`, `get_git_sha_short()`
- `partition_tables_into_groups()`

### `main.py` (~150 lines)
- `cli()` function
- Imports and orchestration only

### `__init__.py`
- Re-export public API for backward compatibility:
```python
from endive.config import configure_from_toml, validate_config, STATS, ...
from endive.catalog import Catalog, AppendCatalog
from endive.transaction import Txn, txn_gen, select_partitions
from endive.conflict import ConflictResolver
from endive.simulation import setup
from endive.main import cli
```

## Global State Strategy

**Challenge**: 50+ globals set by `configure_from_toml()` and accessed throughout.

**Solution**: Keep globals in `config.py`, import where needed.

```python
# In config.py
SIM_SEED: int = None
SIM_DURATION_MS: int = None
N_TABLES: int = None
# ... etc

# In other modules
from endive.config import SIM_SEED, N_TABLES, ...
```

**Why not a Config dataclass?**
- Would require changing every function signature
- Risk of subtle ordering bugs
- Globals with `from endive.config import X` is functionally equivalent

## Execution Plan

### Phase 1: Baseline & Safety Net
1. Record baseline test results with fixed seed
2. Run determinism check: `python -m endive.main config.toml --seed 42` → save output hash
3. Create `endive/errata.md` for bugs found during refactoring

### Phase 2: Extract Utilities (Low Risk)
1. Create `endive/utils.py` with git functions
2. Update imports in main.py
3. Run tests: `pytest tests/ -v`
4. Commit

### Phase 3: Extract Config (Medium Risk)
1. Create `endive/config.py` with:
   - All global declarations
   - `ContentionTracker`
   - Config loading functions
2. Update main.py to import from config
3. Run tests + determinism check
4. Commit

### Phase 4: Extract Latency (Medium Risk)
1. Create `endive/latency.py`
2. Move all latency functions
3. Update imports
4. Run tests + determinism check
5. Commit

### Phase 5: Extract Catalog (Medium Risk)
1. Create `endive/catalog.py`
2. Move `Catalog`, `AppendCatalog`, `LogEntry`
3. Run tests + determinism check
4. Commit

### Phase 6: Extract Conflict (Medium Risk)
1. Create `endive/conflict.py`
2. Move `ConflictResolver`
3. Run tests + determinism check
4. Commit

### Phase 7: Extract Transaction (High Risk)
1. Create `endive/transaction.py`
2. Move `Txn`, commit functions, generation
3. This is the riskiest phase - many interdependencies
4. Run tests + determinism check
5. Commit

### Phase 8: Extract Simulation & Experiment (Medium Risk)
1. Create `endive/simulation.py`
2. Create `endive/experiment.py`
3. Run tests + determinism check
4. Commit

### Phase 9: Finalize main.py (Low Risk)
1. Slim main.py to CLI only
2. Update `__init__.py` with re-exports
3. Run full test suite
4. Run determinism check
5. Final commit

### Phase 10: Verification
1. Run full test suite: `pytest tests/ -v`
2. Run determinism check with multiple seeds
3. Compare experiment output hashes
4. Review errata.md for deferred bugs

### Phase 11: Clean Up Compatibility Shims (Low Risk)
1. Update all test files to import from proper modules
2. Remove re-exports from `main.py`
3. `main.py` contains ONLY CLI code
4. Run tests to verify direct imports work
5. Commit

## Determinism Verification

Before refactoring:
```bash
# Generate baseline
python -m endive.main experiment_configs/single_table_trivial.toml --seed 42 --yes
cp experiments/single_table_trivial-*/42/results.parquet /tmp/baseline.parquet
sha256sum /tmp/baseline.parquet > /tmp/baseline.sha256
```

After each phase:
```bash
# Verify identical output
rm -rf experiments/single_table_trivial-*/42/
python -m endive.main experiment_configs/single_table_trivial.toml --seed 42 --yes
sha256sum experiments/single_table_trivial-*/42/results.parquet
# Must match baseline.sha256
```

## Import Strategy

**Goal**: Tests import directly from appropriate modules.

During refactoring (Phases 2-9), `main.py` temporarily re-exports symbols for safety:
```python
# TEMPORARY - removed in Phase 11
from endive.config import configure_from_toml, ...
```

After Phase 10 verification passes, Phase 11 updates all test imports:
```python
# Before
from endive.main import configure_from_toml, Catalog, Txn

# After
from endive.config import configure_from_toml
from endive.catalog import Catalog
from endive.transaction import Txn
```

No backward compatibility shims retained - this is internal code with no public API.

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Import cycles | Extract in dependency order (utils → config → latency → ...) |
| Global state bugs | Keep globals in single module (config.py) |
| Random seed ordering | Don't reorder any np.random calls |
| Test failures | Run tests after each phase, don't proceed if failing |
| Determinism loss | Hash comparison after each phase |

## Success Criteria

1. All 387 tests pass
2. Determinism verified (same seed → identical parquet)
3. Each module < 600 lines
4. Clear separation of concerns
5. Tests import directly from modules (no indirection)

## Estimated Effort

| Phase | Effort | Risk |
|-------|--------|------|
| 1. Baseline | 10 min | None |
| 2. Utils | 15 min | Low |
| 3. Config | 45 min | Medium |
| 4. Latency | 30 min | Medium |
| 5. Catalog | 30 min | Medium |
| 6. Conflict | 30 min | Medium |
| 7. Transaction | 60 min | High |
| 8. Simulation | 30 min | Medium |
| 9. Finalize | 20 min | Low |
| 10. Verify | 15 min | None |
| 11. Clean imports | 30 min | Low |
| **Total** | **~5.5 hours** | |
