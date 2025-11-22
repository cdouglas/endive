# Analysis Guide: Generating Latency vs Throughput Graphs

This guide explains how to analyze baseline experiment results and generate the latency vs throughput graphs described in ANALYSIS_PLAN.md.

## The Indexing Solution

### Problem

Each parameter combination generates a unique experiment hash, creating directories like:
```
experiments/
├── exp2_1_single_table_false-3c7a944a/   # What parameters?
├── exp2_1_single_table_false-7b31acf3/   # What parameters?
└── exp2_1_single_table_false-9b5cc59e/   # What parameters?
```

**Question:** Which hash corresponds to which `inter_arrival.scale` value?

### Solution

The **`saturation_analysis.py`** module solves this by:

1. **Scanning** experiment directories for pattern matches
2. **Reading** `cfg.toml` from each directory to extract parameters
3. **Loading** all seed results (`seed_dir/results.parquet`)
4. **Aggregating** statistics across multiple seeds
5. **Building** an index mapping hash → parameters → statistics
6. **Plotting** latency vs throughput with proper grouping

## Quick Start

### Analyze Single Experiment

```bash
# Analyze Experiment 2.1 (single table)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Analyze Experiment 2.2 (multi-table)
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables
```

### What Gets Generated

```
plots/exp2_1/
├── experiment_index.csv          # Parameter index for all experiments
├── latency_vs_throughput.png     # Main saturation curve
└── success_rate_vs_load.png      # Success rate and throughput vs load
```

## Understanding the Output

### 1. Experiment Index (`experiment_index.csv`)

Maps each experiment to its parameters and statistics:

| Column | Description |
|--------|-------------|
| `label` | Experiment label (e.g., "exp2_1_single_table_false") |
| `hash` | Unique hash for parameter combination |
| `num_seeds` | Number of seed runs aggregated |
| `inter_arrival_scale` | Inter-arrival time parameter (ms) |
| `num_tables` | Number of tables |
| `real_conflict_probability` | Probability of real conflicts |
| `total_txns` | Total transactions across all seeds |
| `committed` | Successfully committed transactions |
| `success_rate` | Percentage of transactions committed |
| `throughput` | Achieved throughput (commits/sec) |
| `p50/p95/p99_commit_latency` | Latency percentiles (ms) |
| `mean_retries` | Average retry attempts per transaction |

**Example rows:**
```csv
label,hash,inter_arrival_scale,num_tables,success_rate,throughput,p50_commit_latency,p95_commit_latency,p99_commit_latency
exp2_1_single_table_false,fe5c040b,20,1,41.36,56.23,2668.52,7847.34,12677.27
exp2_1_single_table_false,3ef88cc5,50,1,84.43,45.05,2276.21,7251.55,11423.60
exp2_1_single_table_false,f6ea6a90,100,1,99.38,27.22,1328.15,4304.83,7306.76
exp2_1_single_table_false,92cce73c,200,1,100.00,14.78,684.82,2275.48,3674.51
```

### 2. Latency vs Throughput Plot

![Latency vs Throughput Example](latency_vs_throughput.png)

**What it shows:**
- **X-axis:** Achieved throughput (commits/sec)
- **Y-axis:** Commit latency (ms)
- **Lines:** P50, P95, P99 latency percentiles
- **Annotations:** Success rate % at each point
- **Saturation line:** Red dashed line at ~50% success rate

**How to read:**
- **Low load (right side):** High success rate, low latency, but low throughput
- **Medium load (middle):** Balance between success and throughput
- **High load (left side):** Lower success rate, higher latency, throughput plateaus
- **Saturation point:** Where success rate drops below 50%

**Example interpretation:**
```
Point: 56.23 commits/sec, P95=7847ms, Success=41%
→ System is saturated at ~56 commits/sec
→ At saturation, 95% of transactions complete within 7.8 seconds
→ Only 41% of transactions succeed (high abort rate)
```

### 3. Success Rate vs Load Plot

![Success Rate vs Load Example](success_rate_vs_load.png)

**What it shows:**
- **Left panel:** Success rate vs inter-arrival time
- **Right panel:** Throughput vs inter-arrival time
- **Trend:** Lower inter-arrival time = higher load = lower success rate

**How to read:**
- **Low inter-arrival (10-50ms):** High contention, many aborts
- **Medium inter-arrival (100-500ms):** Moderate success, throughput increases
- **High inter-arrival (1000-5000ms):** High success, but low throughput

## Advanced Usage

### Compare Multiple Table Counts

```bash
# Generate plot with separate lines for each table count
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2_grouped \
    --group-by num_tables
```

This creates a plot with multiple lines showing how table count affects saturation.

### Filter by Experiment Label

```bash
# Analyze only specific experiment
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_single_table_false-*" \
    -o plots/exp2_1_only
```

### Custom Output Directory

```bash
# Organize plots by date
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_*" \
    -o plots/$(date +%Y%m%d)
```

## Analysis Workflow

### Step 1: Run Baseline Experiments

```bash
# Start experiments in background
nohup ./run_baseline_experiments.sh --seeds 3 > baseline.log 2>&1 &
```

**Wait time:** ~9.5 hours for full baseline

### Step 2: Verify Results

```bash
# Check experiment directories were created
ls -lh experiments/

# Count result files
find experiments/exp2_* -name "results.parquet" | wc -l

# Expected: 189 for full baseline with 3 seeds
```

### Step 3: Build Index and Generate Plots

```bash
# Experiment 2.1: Single table saturation
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_1_*" \
    -o plots/exp2_1

# Experiment 2.2: Multi-table saturation
python -m icecap.saturation_analysis \
    -i experiments \
    -p "exp2_2_*" \
    -o plots/exp2_2 \
    --group-by num_tables
```

### Step 4: Analyze Results

```bash
# View experiment index
cat plots/exp2_1/experiment_index.csv

# View plots (Linux)
xdg-open plots/exp2_1/latency_vs_throughput.png

# View plots (macOS)
open plots/exp2_1/latency_vs_throughput.png

# View plots (WSL)
explorer.exe plots/exp2_1/
```

### Step 5: Extract Key Findings

From `experiment_index.csv`:

```bash
# Find saturation point (success rate < 55%)
awk -F',' '$10 < 55 {print $4, $11}' plots/exp2_1/experiment_index.csv | \
    sort -k2 -rn | head -n1

# Maximum throughput
awk -F',' 'NR>1 {print $11}' plots/exp2_1/experiment_index.csv | \
    sort -rn | head -n1

# Best latency (at high success rate)
awk -F',' '$10 > 95 {print $4, $14}' plots/exp2_1/experiment_index.csv | \
    sort -k2 -n | head -n1
```

## Programmatic Analysis

For custom analysis, load the index in Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load experiment index
df = pd.read_csv('plots/exp2_1/experiment_index.csv')

# Find saturation point
saturation = df[df['success_rate'] < 55].sort_values('throughput', ascending=False).iloc[0]
print(f"Saturation at {saturation['throughput']:.1f} commits/sec")
print(f"  P95 latency: {saturation['p95_commit_latency']:.0f}ms")
print(f"  Success rate: {saturation['success_rate']:.1f}%")

# Plot custom view
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df['throughput'], df['mean_commit_latency'], s=100, alpha=0.6)
ax.set_xlabel('Throughput (commits/sec)')
ax.set_ylabel('Mean Commit Latency (ms)')
ax.set_title('Custom View: Throughput vs Mean Latency')
plt.savefig('custom_plot.png', dpi=300)
```

## Interpreting Results for Research Questions

### Question 1a: Single Table False Conflicts

**What to find:**
- Maximum throughput before saturation
- P95 latency at saturation
- Success rate curve shape

**Commands:**
```bash
python -m icecap.saturation_analysis -i experiments -p "exp2_1_*" -o plots/exp2_1
cat plots/exp2_1/experiment_index.csv
```

**Key metrics:**
- Saturation throughput (commits/sec at 50% success)
- Latency cost per retry (slope of latency vs retries)
- False conflict resolution cost (~100ms expected)

### Question 2a: Multi-Table False Conflicts

**What to find:**
- How saturation point changes with table count
- Whether more tables → higher throughput
- Impact of multi-table manifest list reads

**Commands:**
```bash
python -m icecap.saturation_analysis \
    -i experiments -p "exp2_2_*" -o plots/exp2_2 --group-by num_tables
```

**Key metrics:**
- Saturation throughput for each table count
- Latency scaling with table count
- Success rate at fixed load with varying tables

## Troubleshooting

### No experiments found

```bash
# Check pattern matches
ls experiments/exp2_*

# Try broader pattern
python -m icecap.saturation_analysis -i experiments -p "exp*" -o plots/all
```

### Missing cfg.toml

```bash
# Verify cfg.toml exists in each experiment directory
find experiments/exp2_* -name "cfg.toml"

# If missing, experiments were not run with experiment label
# Re-run experiments with proper configuration
```

### No seed directories

```bash
# Check for numeric seed directories
find experiments/exp2_1_* -type d -name "[0-9]*"

# Verify results.parquet exists
find experiments/exp2_1_* -name "results.parquet"
```

### Plots look wrong

```bash
# Check experiment index for anomalies
cat plots/exp2_1/experiment_index.csv | less

# Verify parameter extraction
python3 << 'EOF'
import tomli
with open('experiments/exp2_1_single_table_false-HASH/cfg.toml', 'rb') as f:
    config = tomli.load(f)
    print(config['transaction']['inter_arrival']['scale'])
EOF
```

### Dependencies missing

```bash
# Install required packages
pip install pandas matplotlib tomli numpy
```

## Next Steps

After analyzing baseline experiments:

1. **Compare experiments** - Use index to identify trends
2. **Run Phase 3** - Real conflict experiments (to be implemented)
3. **Generate paper figures** - Use plots in writeup
4. **Identify optimal settings** - Based on throughput/latency tradeoff

## Reference

- **ANALYSIS_PLAN.md** - Complete research plan and methodology
- **RUNNING_EXPERIMENTS.md** - How to run baseline experiments
- **experiment_configs/README.md** - Configuration details
- **icecap/saturation_analysis.py** - Implementation source code
