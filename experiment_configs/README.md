# Experiment Configuration Files

This directory contains configuration files for the experiments described in `ANALYSIS_PLAN.md`.

## Quick Start

Each configuration file is ready to run and includes:
- Detailed comments explaining the experiment setup
- Expected results
- Parameters to sweep for full analysis
- Example commands

### Running a Single Experiment

```bash
# Activate virtual environment
source bin/activate

# Run with default parameters
python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml
```

### Running Parameter Sweeps

Each experiment specifies parameters to sweep. For example, to sweep offered load:

```bash
# Create a copy and modify the inter_arrival.scale parameter
for scale in 10 20 50 100 200 500 1000 2000 5000; do
    sed "s/inter_arrival.scale = .*/inter_arrival.scale = $scale/" \
        experiment_configs/exp2_1_single_table_false_conflicts.toml > temp.toml
    echo "Y" | python -m icecap.main temp.toml
done
```

### Running Multiple Seeds

To average results across multiple runs:

```bash
# Run same config 10 times with different random seeds
for i in {1..10}; do
    echo "Y" | python -m icecap.main experiment_configs/exp2_1_single_table_false_conflicts.toml
done
```

## Experiment Descriptions

### Phase 2: Baseline Experiments (False Conflicts Only)

#### Experiment 2.1: Single Table Saturation
**File:** `exp2_1_single_table_false_conflicts.toml`

**Research Question:** What is the maximum throughput for a single table with false conflicts only?

**Key Parameters to Sweep:**
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

**Key Findings:**
- Saturation point (abort rate > 50%)
- Latency vs throughput curve
- Cost breakdown: CAS vs manifest lists

#### Experiment 2.2: Multi-Table Saturation
**File:** `exp2_2_multi_table_false_conflicts.toml`

**Research Question:** How does table count affect maximum throughput?

**Key Parameters to Sweep:**
- `num_tables` (and `num_groups`): [1, 2, 5, 10, 20, 50]
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

**Note:** Always set `num_groups = num_tables` for table-level conflicts

**Key Findings:**
- More tables → less contention
- But multi-table transactions have higher cost
- Optimal table count for workload

### Phase 3: Real Conflict Experiments

#### Experiment 3.1: Real Conflict Impact
**File:** `exp3_1_single_table_real_conflicts.toml`

**Research Question:** How do real conflicts (requiring manifest file operations) affect saturation?

**Key Parameters to Sweep:**
- `real_conflict_probability`: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

**Key Findings:**
- Cost difference: false (~100ms) vs real (~500ms+) conflicts
- Saturation point shifts with real conflict probability
- Maximum throughput reduction

#### Experiment 3.2: Manifest Count Distribution
**File:** `exp3_2_manifest_count_distribution.toml`

**Research Question:** How does variance in conflicting manifest count affect performance?

**Key Parameters to Sweep:**
- `conflicting_manifests.distribution`: ["fixed", "exponential", "uniform"]
- `conflicting_manifests.value`: [1, 5, 10] (for fixed)
- `conflicting_manifests.mean`: [1, 3, 5, 10] (for exponential)
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

**Key Findings:**
- Impact of worst-case scenarios (many conflicting manifests)
- Effect of variance (exponential vs fixed distributions)
- Parallelism limit interaction (max_parallel = 4)

#### Experiment 3.3: Multi-Table Real Conflicts
**File:** `exp3_3_multi_table_real_conflicts.toml`

**Research Question:** How do real conflicts interact with table count?

**Key Insight:** P(≥1 real conflict) = 1 - (1 - p)^n_tables (compounds!)

**Key Parameters to Sweep:**
- `num_tables` (and `num_groups`): [1, 2, 5, 10, 20]
- `real_conflict_probability`: [0.0, 0.1, 0.3, 0.5]
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

**Key Findings:**
- Non-linear interaction between table count and real conflicts
- Heatmap of saturation throughput
- Optimal table count depends on conflict probability

## Output Organization

All experiments use the `experiment.label` parameter to organize outputs:

```
experiments/
├── exp2_1_single_table_false-a1b2c3d4/
│   ├── cfg.toml                    # Saved configuration
│   ├── 12345/results.parquet       # Run with seed 12345
│   ├── 67890/results.parquet       # Run with seed 67890
│   └── ...
├── exp3_1_single_table_real-e5f6g7h8/
│   ├── cfg.toml
│   ├── ...
```

The hash (e.g., `a1b2c3d4`) is deterministic based on:
- All config parameters (except seed and label)
- Simulator code (all `icecap/*.py` files)

This means:
- Same config + code → same hash (easy to find related runs)
- Different parameters → different hash (prevents mixing results)
- Code changes → different hash (tracks simulator version)

## Analysis

After running experiments, use the analysis tools:

```bash
# Generate all plots for an experiment
python -m icecap.analysis all \
    -i experiments/exp2_1_single_table_false-* \
    -o plots/exp2_1

# Generate specific plots
python -m icecap.analysis cdf -i experiments/exp2_1_* -o plots/exp2_1
python -m icecap.analysis success-rate -i experiments/exp2_1_* -o plots/exp2_1

# Generate summary table
python -m icecap.analysis summary -i experiments/exp2_1_* -o plots/exp2_1
```

## Tips

### Parameter Sweep Script

Create a helper script to automate parameter sweeps:

```bash
#!/bin/bash
# sweep_experiment.sh

CONFIG_FILE=$1
PARAM_NAME=$2
shift 2
PARAM_VALUES=("$@")

for value in "${PARAM_VALUES[@]}"; do
    echo "Running with $PARAM_NAME = $value"
    sed "s/$PARAM_NAME = .*/$PARAM_NAME = $value/" "$CONFIG_FILE" > temp.toml
    echo "Y" | python -m icecap.main temp.toml
done

rm temp.toml
```

Usage:
```bash
chmod +x sweep_experiment.sh
./sweep_experiment.sh \
    experiment_configs/exp2_1_single_table_false_conflicts.toml \
    "inter_arrival.scale" \
    10 20 50 100 200 500 1000 2000 5000
```

### Automated Analysis

After running all sweeps for an experiment, analyze results:

```bash
#!/bin/bash
# analyze_experiment.sh

EXPERIMENT_PATTERN=$1
OUTPUT_DIR=$2

mkdir -p "$OUTPUT_DIR"

# Generate all plots
python -m icecap.analysis all \
    -i experiments/$EXPERIMENT_PATTERN \
    -o "$OUTPUT_DIR"

# Print summary
echo "=== Experiment Summary ==="
cat "$OUTPUT_DIR/summary.csv"
```

Usage:
```bash
./analyze_experiment.sh "exp2_1_single_table_false-*" plots/exp2_1
```

## Validation

Before running full experiment sweeps, validate with quick tests:

```bash
# Short duration for quick validation
sed 's/duration_ms = .*/duration_ms = 10000/' \
    experiment_configs/exp2_1_single_table_false_conflicts.toml > test.toml

echo "Y" | python -m icecap.main test.toml -v

# Check that:
# - Configuration loads correctly
# - Experiment label creates proper directory structure
# - Statistics include conflict type breakdown
# - Results are exported successfully
```

## Next Steps

1. **Run baseline experiments** (Phase 2)
   - Start with Experiment 2.1 (single table, false conflicts)
   - Sweep offered load to find saturation point
   - Run multiple seeds for statistical confidence

2. **Run real conflict experiments** (Phase 3)
   - Experiment 3.1: Compare false vs real conflicts
   - Experiment 3.2: Study variance in conflict cost
   - Experiment 3.3: Multi-table interaction effects

3. **Analyze results**
   - Generate latency vs throughput curves
   - Identify saturation points
   - Create comparison plots
   - Generate summary tables

4. **Document findings**
   - Update ANALYSIS_PLAN.md with results
   - Create visualizations
   - Write recommendations

## References

- `ANALYSIS_PLAN.md`: Complete analysis plan with research questions
- `README.md`: Simulator overview and configuration guide
- `QUICKSTART.md`: Getting started with the simulator
- `tests/test_conflict_types.py`: Tests for false vs real conflict functionality
- `tests/test_experiment_structure.py`: Tests for experiment organization
