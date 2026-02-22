# Documentation Index

This index catalogs the key insights from the documentation and codebase analysis performed during the refactoring specification work.

## Quick Navigation

| Document | Purpose | Key Sections |
|----------|---------|--------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design patterns, invariants, code locations | Critical Design Patterns, Key Invariants |
| [DEVELOPER_NOTES.md](DEVELOPER_NOTES.md) | Quick reference, common issues | Token-Saving Strategies, Gotchas |
| [APPENDIX_SIMULATOR_DETAILS.md](APPENDIX_SIMULATOR_DETAILS.md) | Probability distributions, formulas | Cost Model, Latency Distributions |
| [WARMUP_PERIOD.md](WARMUP_PERIOD.md) | Warmup methodology | Formula, Validation |
| [SNAPSHOT_VERSIONING.md](SNAPSHOT_VERSIONING.md) | Version tracking verification | CAS Semantics, Manifest List Reading |
| [CONSOLIDATED_FORMAT.md](CONSOLIDATED_FORMAT.md) | Parquet file structure | Predicate Pushdown, Row Groups |
| [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md) | Plot generation, interpretation | Saturation Curves, Filtering |
| [QUICKSTART.md](QUICKSTART.md) | Installation, first simulation | Getting Started |
| [DOCKER.md](DOCKER.md) | Container execution | EXP_ARGS, Volume Mounts |

---

## Critical Invariants

These MUST be preserved in any refactoring:

### 1. Version Monotonicity
**Location**: `endive/main.py:359-393`

```
catalog.seq advances by exactly 1 on each successful commit.
Never decreases. Never skips values.
```

**Why**: Breaking this invalidates the snapshot versioning scheme entirely.

### 2. Manifest List Reading Exactness
**Location**: `endive/main.py:424-452`

```
When N snapshots behind, read EXACTLY N manifest lists.
Not N-1, not N+1.
```

**Why**: Part of Iceberg's validationHistory() protocol.

### 3. Transaction-Catalog Message Passing
**Location**: `endive/snapshot.py`

```
Transactions NEVER access Catalog state directly.
All state obtained via immutable CatalogSnapshot.
```

**Why**: Ensures latency costs are explicit and state is captured at specific time.

### 4. False vs Real Conflict Distinction
**Location**: `endive/conflict.py`, `endive/operation.py`

```
False conflicts: Different partitions, ~100ms cost
Real conflicts: Same partition, ~400ms+ cost (may abort)
```

**Why**: The cost difference is critical for performance analysis.

### 5. Seed Determinism
**Location**: `endive/main.py:220, 1078-1085`

```
Same seed + same code = bitwise identical results.
All randomness via np.random with seed control.
```

**Why**: Reproducibility for validation and debugging.

---

## Key Data Sources

### YCSB Benchmark Data
**Location**: `analysis/distributions.json`

Contains empirical latency measurements for all storage providers:
- CAS success/failure latencies
- Append latencies
- GET/PUT latencies by size

```python
# Extract min latencies
import json
with open('analysis/distributions.json') as f:
    data = json.load(f)
# data['AWS_S3_Express']['CAS']['per_thread']['1']['latency_ms']['min']
```

### Provider Profiles
**Location**: `endive/config.py:72-273` (PROVIDER_PROFILES dict)

Each profile includes:
- CAS latencies (success, failure)
- Append latencies
- PUT latency model (base + rate * size)
- Contention scaling factors

### Cost Model Summary

| Operation | S3 | S3X | Azure | Instant |
|-----------|-----|-----|-------|---------|
| CAS (median) | 61ms | 22ms | 93ms | 1ms |
| False conflict | ~160ms | ~60ms | ~200ms | ~5ms |
| Real conflict | ~450ms+ | ~200ms+ | ~550ms+ | ~20ms+ |

---

## Module Structure (Current)

```
endive/
├── main.py           # Monolith: 3263 lines, 8+ concerns
├── config.py         # Configuration, provider profiles
├── snapshot.py       # CatalogSnapshot, CASResult
├── transaction.py    # Txn, LogEntry, operation_type
├── operation.py      # OperationType enum, behaviors
├── conflict.py       # ConflictResolverV2
├── capstats.py       # Statistics collection
├── saturation_analysis.py  # Analysis pipeline
├── experiment.py     # Experiment runner helpers
└── test_utils.py     # Test configuration builders
```

### Key Functions in main.py

| Function | Lines | Purpose |
|----------|-------|---------|
| `configure_from_toml()` | 746-1070 | Config loading |
| `txn_gen()` | 483-560 | Transaction generator |
| `txn_commit()` | 560-650 | CAS commit protocol |
| `Catalog.try_cas()` | 1850-1920 | CAS operation |
| `resolve_conflict()` | 627-719 | Conflict resolution |
| `get_cas_latency()` | 1251-1300 | Latency generation |

---

## Global Variables (Current)

These will be eliminated in the refactor:

### Simulation Control
- `SIM_DURATION_MS`, `SIM_OUTPUT_PATH`, `SIM_SEED`

### Catalog Configuration
- `N_TABLES` (fixed at construction in new design)

### Storage Latencies
- `T_CAS`, `T_APPEND`, `T_MANIFEST_LIST`, etc.
- **Replacement**: `LatencyDistribution` objects in `StorageProvider`

### Transaction Parameters
- `T_RUNTIME_MU`, `T_RUNTIME_SIGMA`, `INTER_ARRIVAL_PARAMS`
- **Replacement**: `WorkloadConfig` with distribution objects

### Conflict Parameters
- `REAL_CONFLICT_PROBABILITY`, `CONFLICTING_MANIFESTS_DIST`
- **Replacement**: `ConflictDetector` configuration

### Contention
- `CONTENTION_TRACKER`, `MAX_PARALLEL`
- **Replacement**: Internal to `StorageProvider`

---

## Test Categories

### Tier 1: Direct Transfer (100% reusable)
- `test_exponential_backoff.py` - Algorithm correctness
- `test_numerical_accuracy.py` - Mathematical invariants
- `test_statistical_rigor.py` - Distribution properties

### Tier 2: Adapt API Calls
- `test_simulator.py` - Update configuration builder
- `test_conflict_resolution.py` - Use new conflict detector
- `test_snapshot_versioning.py` - Use new catalog interface

### Tier 3: Rewrite
- Tests directly manipulating internal state
- Tests using `partition_tables_into_groups` (removed)

---

## Common Patterns

### Latency Generation (Current)
```python
# Scattered conditional checks
if STORAGE_PROVIDER_INSTANCE:
    latency = STORAGE_PROVIDER_INSTANCE.get_cas_latency()
else:
    latency = generate_latency_from_config(T_CAS)
```

### Latency Generation (New)
```python
# Always use opaque distribution
latency = self._storage.cas_latency.sample()
```

### Configuration Loading (Current)
```python
global N_TABLES, T_CAS, ...
N_TABLES = config['catalog']['num_tables']
```

### Configuration Loading (New)
```python
# Returns configured object, no globals
catalog = CASCatalog(storage, num_tables=config['catalog']['num_tables'])
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Simulation speed | ~1000x real-time |
| 1-hour simulation | ~3.6 seconds |
| Memory per simulation | ~100MB |
| Full test suite | ~3 minutes |
| Full experiment suite | ~24 hours (8 cores) |

---

## Experiment Organization

```
experiments/
├── {label}-{hash}/           # Deterministic from parameters
│   ├── cfg.toml              # Configuration snapshot
│   ├── {seed}/results.parquet
│   └── {seed}/results.parquet
└── consolidated.parquet      # All experiments (v2.0+)
```

**Hash computation**: `compute_experiment_hash()` in `main.py:692-745`

---

## Analysis Pipeline

```
1. build_experiment_index()   # Scan experiments/
2. load_and_aggregate_results()  # Load parquet files
3. compute_warmup_duration()  # Filter transients
4. compute_aggregate_statistics()  # Calculate metrics
5. plot_*()                   # Generate figures
```

**Warmup formula**:
```python
warmup_ms = max(5 * 60 * 1000, min(3 * mean_runtime, 15 * 60 * 1000))
```

---

## Refactoring Checkpoints

After each phase, verify:

1. **Determinism**: Same seed produces identical output hash
2. **Invariants**: All version/conflict invariants hold
3. **Tests**: Full test suite passes
4. **Performance**: No significant slowdown

---

## Known Limitations

### Accurately Modeled
- Optimistic concurrency control with CAS
- Snapshot isolation semantics
- Manifest list reading on conflict
- False vs real conflict distinction
- Stochastic storage latencies
- Parallel I/O with limits

### Simplified
- Actual data merging (latency only)
- Manifest file structure (not explicit)
- Network failures (no transient errors)
- Read queries (write-only transactions)

### Cannot Measure
- Per-operation timing breakdown
- Per-table I/O in multi-table transactions
- Backoff wait time separately

---

## References

- Apache Iceberg specification
- VLDB 2023 paper on cloud storage latencies (Durner et al.)
- YCSB benchmark methodology (June 2025 data)
