# Consolidated Experiment Results Format

## Overview

`scripts/consolidate.py` writes a single parquet file containing every experiment's transactions, in addition to the original per-seed `results.parquet` files. The consolidated file gives the analysis pipeline efficient predicate-pushdown access without loading thousands of small files.

```
experiments/
├── consolidated.parquet             # Single merged file (all experiments)
├── exp1_fa_baseline-<hash>/
│   ├── cfg.toml                     # Configuration snapshot (preserved)
│   ├── version.txt                  # Git SHA at run time
│   ├── 42/results.parquet           # Seed 42 (preserved)
│   └── 43/results.parquet           # Seed 43 (preserved)
└── exp2_mix_heatmap-<hash>/
    └── ...
```

Original per-seed files are preserved so the consolidated file can be regenerated at any time.

## Schema

The consolidated schema is discovered dynamically from the per-seed `results.parquet` files by `consolidate.py:discover_schema()`, taking the union of columns across all experiments. Every row carries:

| Column | Description |
|--------|-------------|
| `exp_name` | Experiment label (e.g., `exp1_fa_baseline`) |
| `exp_hash` | Deterministic hash of the parameters for this run |
| `seed` | Random seed for the run |
| `config` | Flattened TOML config as a `map<string, string>` |

plus the transaction columns defined in [SPEC.md §6.2](../SPEC.md) (`txn_id`, `t_submit`, `t_runtime`, `t_commit`, `commit_latency`, `total_latency`, `n_retries`, `status`, `operation_type`, `abort_reason`, the per-phase timing decomposition, and the I/O counters).

**Sort order**: `(exp_name, exp_hash, seed, t_submit)`.
**Row groups**: one per seed, enabling predicate pushdown to skip entire seeds at the storage layer.
**Compression**: ZSTD level 3 (pyarrow default).

## Using the Consolidated File

`endive/saturation_analysis.py` uses the consolidated file automatically when present, and falls back to individual `results.parquet` files otherwise. For ad-hoc access:

```python
import pandas as pd

df = pd.read_parquet(
    'experiments/consolidated.parquet',
    filters=[
        ('exp_name', '==', 'exp1_fa_baseline'),
        ('exp_hash', '==', '7fe68106'),
        ('t_submit', '>=', 900_000),    # After warmup
        ('t_submit', '<',  2_700_000),  # Before cooldown
    ],
)
```

Always supply filters — loading the full file pulls every experiment's rows into memory.

## Regeneration

```bash
# Rebuild from the per-seed results.parquet files
python scripts/consolidate.py

# Also supports writing one consolidated file per experiment label
python scripts/consolidate.py --partition
```

`consolidate.py` streams row groups to disk incrementally, so memory stays bounded even with hundreds of experiments. See `scripts/consolidate.py` for the full flag set.
