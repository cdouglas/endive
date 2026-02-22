# Experiment Plan: Operation Types and Catalog Latency

## Overview

With accurate operation type modeling (FastAppend, MergeAppend, ValidatedOverwrite) and partition-aware conflict resolution, we can now run meaningful experiments that reflect Iceberg's actual behavior.

---

## End-to-End Validation

Before running full experiments, validate the pipeline from simulation → parquet → analysis.

### V1: Single Run Smoke Test

```bash
# Run minimal simulation
python -m endive.main experiment_configs/baseline_instant.toml --yes

# Verify parquet output
python -c "
import pandas as pd
df = pd.read_parquet('experiments/baseline_instant-*/42/results.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'operation_type values: {df[\"operation_type\"].value_counts().to_dict()}')
print(f'status values: {df[\"status\"].value_counts().to_dict()}')
print(f'abort_reason values: {df[\"abort_reason\"].value_counts().to_dict()}')
"
```

**Expected**:
- `operation_type` column populated (not all NaN)
- `abort_reason` shows "validation_exception" for VO aborts, "max_retries" for retry exhaustion

### V2: Operation Type Distribution Validation

```bash
# Run with 50/50 mix
# Create test config with operation_types.fast_append = 0.5, validated_overwrite = 0.5

python -c "
import pandas as pd
df = pd.read_parquet('experiments/test_mix-*/42/results.parquet')
fa = len(df[df['operation_type'] == 'fast_append'])
vo = len(df[df['operation_type'] == 'validated_overwrite'])
print(f'FastAppend: {fa} ({100*fa/len(df):.1f}%)')
print(f'ValidatedOverwrite: {vo} ({100*vo/len(df):.1f}%)')
# Should be approximately 50/50
"
```

### V3: Conflict Resolution Validation

```bash
# Run with 100% ValidatedOverwrite, real_conflict_probability = 1.0
# Every conflict should abort with ValidationException

python -c "
import pandas as pd
df = pd.read_parquet('experiments/test_vo_abort-*/42/results.parquet')
aborted = df[df['status'] == 'aborted']
ve_aborts = len(aborted[aborted['abort_reason'] == 'validation_exception'])
mr_aborts = len(aborted[aborted['abort_reason'] == 'max_retries'])
print(f'ValidationException aborts: {ve_aborts}')
print(f'MaxRetries aborts: {mr_aborts}')
# With real_conflict_probability=1.0, all aborts should be validation_exception
# (unless transaction never conflicted, which means it committed)
"
```

### V4: Per-Operation-Type Analysis

```bash
# Verify we can compute metrics per operation type
python -c "
import pandas as pd
df = pd.read_parquet('experiments/test_mix-*/42/results.parquet')

for op_type in ['fast_append', 'validated_overwrite']:
    subset = df[df['operation_type'] == op_type]
    committed = subset[subset['status'] == 'committed']
    success_rate = len(committed) / len(subset) * 100 if len(subset) > 0 else 0
    p99 = committed['commit_latency'].quantile(0.99) if len(committed) > 0 else float('nan')
    print(f'{op_type}: {len(subset)} txns, {success_rate:.1f}% success, P99={p99:.0f}ms')
"
```

### V5: Heatmap Data Structure

```bash
# Verify we can aggregate across (mix_ratio, arrival_rate) dimensions
python -c "
import pandas as pd
import glob

# Load all experiments matching pattern
results = []
for path in glob.glob('experiments/exp_mix_*_arr_*/*/results.parquet'):
    df = pd.read_parquet(path)
    # Extract parameters from path or cfg.toml
    # Compute metrics per (mix, arrival_rate) cell
    pass

# Output should be pivotable into heatmap
"
```

---

## Proposed Experiments

### Experiment 1: FastAppend Baseline (Instant Catalog)

**Goal**: Establish baseline throughput with cheapest operation type and instant catalog.

**Setup**:
- Catalog: Instant (1ms CAS)
- Storage: S3 latencies
- Operation types: 100% FastAppend
- Single table
- Sweep arrival rates: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms

**Metrics**: Throughput, success rate, P50/P95/P99 latency, mean retries

**Validation checkpoints**:
- [ ] All `operation_type` values are `fast_append` or `None` (legacy)
- [ ] No `validation_exception` aborts (FastAppend cannot have real conflicts)
- [ ] Success rate monotonically decreases as arrival rate decreases (higher load)

**Analysis**:
- Line plot: success rate vs throughput
- Line plot: P50/P95/P99 latency vs throughput
- Identify saturation point (where success rate drops below 100%)

---

### Experiment 2: FastAppend/ValidatedOverwrite Mix (Heatmap)

**Goal**: Understand how maintenance operations (compaction, MERGE INTO) compete with streaming appends.

**Setup**:
- Catalog: Instant (1ms CAS)
- Storage: S3 latencies
- Operation mix sweep: fast_append = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0]
                       validated_overwrite = 1.0 - fast_append
- Arrival rate sweep: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms
- Single table
- **Real conflict probability: 0.1** (10% of VO conflicts are data overlap → abort)

**Heatmaps** (X: arrival rate, Y: VO fraction):
1. VO transaction success rate
2. FastAppend transaction success rate (for comparison)
3. Overall mean commit latency
4. VO P99 commit latency
5. Validation exception count (absolute or as % of VO transactions)

**Validation checkpoints**:
- [ ] Operation type distribution matches config (within sampling variance)
- [ ] `validation_exception` aborts only occur for `validated_overwrite` transactions
- [ ] At 100% FastAppend row, no validation exceptions
- [ ] At 100% VO row with low arrival rate, some transactions succeed

**Expected patterns**:
- VO success rate degrades faster than FastAppend as load increases
- At high FastAppend ratio (0.9), VO transactions face massive I/O convoy
- FastAppend success rate relatively stable across mix ratios (no I/O convoy)

---

### Experiment 3a: Catalog Latency Sweep (100% FastAppend)

**Goal**: Determine catalog latency impact on pure append workload.

**Setup**:
- Catalog latency sweep: [1, 15, 50, 100, 500] ms (instant, S3X, Azure Premium, S3/Azure, GCP)
- Storage: S3 latencies (constant)
- Operation types: 100% FastAppend
- Single table
- Arrival rate sweep: [50, 100, 200, 500] ms (subset of Exp 1)

**Provider reference latencies**:
| Provider | CAS Latency (median) | Config |
|----------|---------------------|--------|
| Instant | 1ms | `provider = "instant"` |
| S3 Express | ~15ms | `T_CAS.mean = 15` |
| Azure Premium | ~50ms | `provider = "azure_premium"` |
| S3 / Azure | ~100ms | `provider = "s3"` |
| GCP | ~500ms | `provider = "gcp"` |

**Validation checkpoints**:
- [ ] Higher catalog latency → lower throughput at same arrival rate
- [ ] Effect is moderate (catalog adds ~1 RTT per attempt, vs ~160ms storage I/O)

**Analysis**:
- Line plot: success rate vs throughput, grouped by catalog latency
- Quantify: % throughput reduction per 100ms catalog latency increase

---

### Experiment 3b: Catalog Latency Sweep (90/10 Mix)

**Goal**: Determine catalog latency impact when maintenance operations are present.

**Setup**:
- Catalog latency sweep: [1, 15, 50, 100, 500] ms
- Storage: S3 latencies (constant)
- Operation types: 90% FastAppend, 10% ValidatedOverwrite
- Real conflict probability: 0.1
- Single table
- Arrival rate sweep: [50, 100, 200, 500] ms

**Validation checkpoints**:
- [ ] VO success rate degrades more than FastAppend as catalog latency increases
- [ ] Validation exceptions present (VO can have real conflicts)
- [ ] Compare to Exp 3a: mixed workload more sensitive to catalog latency?

**Analysis**:
- Line plot: VO success rate vs throughput, grouped by catalog latency
- Line plot: FastAppend success rate vs throughput, grouped by catalog latency
- Compare slopes: does catalog latency hurt VO more than FastAppend?

**Hypothesis**: Catalog latency has similar impact on both operation types because:
- Catalog latency affects CAS attempt (both types)
- I/O convoy is storage latency, not catalog latency
- But: higher catalog latency → more time for other txns to commit → higher n_behind → larger I/O convoy

---

## Recommended Follow-up Experiments

### Experiment 1b: MergeAppend Baseline

**Rationale**: MergeAppend is between FastAppend and ValidatedOverwrite:
- No I/O convoy (no historical ML reads)
- But manifest file re-merge cost scales with n_behind

**Setup**: Same as Exp 1, but 100% MergeAppend with `manifests_per_concurrent_commit = 1.0`

### Experiment 4: Partition Mode

**Rationale**: With partitions, conflicts are per-partition, not per-table.

**Setup**:
- Enable partition mode
- Sweep num_partitions: [1, 4, 16, 64, 256]
- Mixed workload (90/10 FastAppend/VO)
- Real conflict probability: 0.1

**Hypothesis**: More partitions → fewer conflicts → higher throughput, but VO still pays per-partition ML read cost for overlapping partitions

---

## Implementation Checklist

### Config Templates

```toml
# experiment_configs/exp1_fastappend_baseline.toml
[simulation]
duration_ms = 3600000  # 1 hour
seed = 42

[catalog]
num_tables = 1
num_groups = 1

[storage]
provider = "instant"  # 1ms CAS

[transaction]
operation_types.fast_append = 1.0
operation_types.validated_overwrite = 0.0
inter_arrival.scale = 100  # Sweep this
```

```toml
# experiment_configs/exp2_mix_sweep.toml
[transaction]
operation_types.fast_append = 0.9  # Sweep this
operation_types.validated_overwrite = 0.1
real_conflict_probability = 0.1
inter_arrival.scale = 100  # Sweep this
```

```toml
# experiment_configs/exp3a_catalog_latency_fa.toml
[storage]
provider = "s3"  # Sweep: instant, s3x, azure_premium, s3, gcp

[transaction]
operation_types.fast_append = 1.0
```

```toml
# experiment_configs/exp3b_catalog_latency_mix.toml
[storage]
provider = "s3"  # Sweep: instant, s3x, azure_premium, s3, gcp

[transaction]
operation_types.fast_append = 0.9
operation_types.validated_overwrite = 0.1
real_conflict_probability = 0.1
```

### Analysis Scripts Needed

1. **Per-operation-type metrics**: Filter parquet by `operation_type`, compute success rate / latency / abort reasons per type

2. **Heatmap generation**:
   - Input: experiment results across (param1, param2) grid
   - Output: 2D heatmap PNG with color scale

3. **Validation exception analysis**:
   - Count and % of VO transactions that abort with ValidationException
   - Compare to retry-exhaustion aborts

### Experiment Runner Updates

Add to `scripts/run_all_experiments.py`:

```python
EXPERIMENT_GROUPS = {
    # ... existing groups ...

    "exp1_fa_baseline": {
        "config": "experiment_configs/exp1_fastappend_baseline.toml",
        "params": {"inter_arrival.scale": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]},
    },

    "exp2_mix": {
        "config": "experiment_configs/exp2_mix_sweep.toml",
        "params": {
            "transaction.operation_types.fast_append": [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.0],
            "inter_arrival.scale": [10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        },
    },

    "exp3a_catalog_fa": {
        "config": "experiment_configs/exp3a_catalog_latency_fa.toml",
        "params": {
            "storage.provider": ["instant", "s3x", "azure_premium", "s3", "gcp"],
            "inter_arrival.scale": [50, 100, 200, 500],
        },
    },

    "exp3b_catalog_mix": {
        "config": "experiment_configs/exp3b_catalog_latency_mix.toml",
        "params": {
            "storage.provider": ["instant", "s3x", "azure_premium", "s3", "gcp"],
            "inter_arrival.scale": [50, 100, 200, 500],
        },
    },
}
```

---

## Questions Resolved

1. **real_conflict_probability**: Set to 0.1 (10%) - low enough that most VO conflicts are false conflicts requiring merge, but some abort with ValidationException

2. **Catalog latency experiments**: Split into 3a (pure FastAppend) and 3b (90/10 mix) to isolate effects

3. **Validation**: Added comprehensive end-to-end validation section with specific checkpoints per experiment
