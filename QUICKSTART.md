# Quick Start Guide

## Setup

```bash
# Activate virtual environment
source bin/activate

# Verify installation
python -m icecap.main --help
```

## Understanding the Simulator

Icecap models Apache Iceberg-style optimistic concurrency control with:
- **Compare-and-swap (CAS)** for atomic catalog updates
- **Manifest list reading** when transactions fall behind (reads n lists for n snapshots)
- **Stochastic storage latencies** using normal distributions
- **Configurable parallelism** for manifest operations during conflict resolution

## Run Your First Simulation

### Single Simulation

```bash
# Run with default configuration
python -m icecap.main cfg.toml

# The simulator will:
# 1. Display configuration summary
# 2. Ask for confirmation (press Enter or 'y')
# 3. Show progress bar
# 4. Export results to Parquet file
# 5. Display summary statistics
```

**Example output:**
```
[Storage Latencies (ms, mean±stddev)]
  Min Latency:  5.0ms
  CAS:          100.0±10.0
  Metadata R/W: 100.0±10.0 / 120.0±15.0
  Manifest L:   100.0±10.0 / 120.0±15.0
  Manifest F:   100.0±10.0 / 120.0±15.0
  Max Parallel: 4

Simulation Summary:
  Total transactions: 20083
  Committed: 19856 (98.9%)
  Aborted: 227 (1.1%)
  Commit latency (ms):
    Mean: 234.56
    Median: 210.00
    P95: 485.20
    P99: 892.45
  Retries per transaction:
    Mean: 0.87
    Max: 8
```

### Automated Mode

For scripting or batch processing:

```bash
# Skip confirmation and progress bar
python -m icecap.main cfg.toml -y --no-progress -q
```

## Run Experiment Sweeps

### 1. Explore Client Load Impact

Generate CDF and success rate plots for different client loads:

```bash
# Run experiments with varying inter-arrival times
# Lower time = more concurrent clients = higher contention
python -m icecap.experiment -o experiments/clients sweep-clients \
    --times 100 500 1000 2000 5000 10000 \
    --dist exponential

# Generate all plots
python -m icecap.analysis -i experiments/clients -o plots/clients all
```

**Outputs:**
- `cdf_commit_latency_clients.png` - Latency distributions by load
- `success_rate_clients.png` - Abort rates and throughput
- `summary.csv` - Detailed statistics for all experiments

**What to look for:**
- **100ms inter-arrival**: High load, many conflicts, higher latency tail
- **10000ms inter-arrival**: Low load, few conflicts, tighter latency distribution
- **Optimal point**: Where throughput peaks before aborts increase dramatically

### 2. Explore Catalog Latency Impact

Test how storage performance affects commit times:

```bash
# Run experiments with different CAS latencies
python -m icecap.experiment -o experiments/latency sweep-latency \
    --latencies 10 50 100 200 500 1000

# Generate plots
python -m icecap.analysis -i experiments/latency -o plots/latency all
```

**What to look for:**
- Higher CAS latency amplifies retry costs exponentially
- Manifest list reads on conflict multiply the latency impact
- Fast storage (10-50ms) vs slow storage (500-1000ms) dramatically affects tail latency

### 3. Combined Analysis

Full factorial sweep of client load × catalog latency:

```bash
# 9 experiments: 3 loads × 3 latencies
python -m icecap.experiment -o experiments/combined sweep-combined \
    --times 500 1000 5000 \
    --latencies 50 100 200

# Generate comprehensive analysis
python -m icecap.analysis -i experiments/combined -o plots/combined all
```

**Outputs:**
- `catalog_latency_impact.png` - 4-panel analysis showing:
  - Mean commit latency vs CAS latency
  - P95 commit latency vs CAS latency
  - Retry rate vs CAS latency
  - Success rate vs CAS latency
  - Each line represents different client load

**Insights:**
- Interaction effects between load and latency
- Non-linear scaling at high contention
- Optimal operating regions for different storage systems

## Customize Configuration

### Basic Configuration

Edit `cfg.toml`:

```toml
[simulation]
duration_ms = 100000000  # 100 seconds of simulated time
output_path = "results.parquet"
seed = 42                # For reproducible results (omit for random)

[catalog]
num_tables = 10          # More tables = less contention

[transaction]
retry = 10               # Max retry attempts before abort
inter_arrival.distribution = "exponential"
inter_arrival.scale = 5000.0  # Mean ~0.2 txn/sec
```

### Storage Configuration (Key Feature)

Configure realistic storage latencies with stochastic variance:

```toml
[storage]
# Parallelism limit for manifest operations during conflict resolution
max_parallel = 4

# Minimum latency prevents unrealistic zeros from normal distribution
min_latency = 5

# CAS operation latency (ms) - normal distribution
T_CAS.mean = 100
T_CAS.stddev = 10        # ~10% coefficient of variation

# Metadata root operations (read typically faster than write)
T_METADATA_ROOT.read.mean = 100
T_METADATA_ROOT.read.stddev = 10
T_METADATA_ROOT.write.mean = 120
T_METADATA_ROOT.write.stddev = 15

# Manifest list operations
T_MANIFEST_LIST.read.mean = 100
T_MANIFEST_LIST.read.stddev = 10
T_MANIFEST_LIST.write.mean = 120
T_MANIFEST_LIST.write.stddev = 15

# Manifest file operations (often largest objects)
T_MANIFEST_FILE.read.mean = 100
T_MANIFEST_FILE.read.stddev = 10
T_MANIFEST_FILE.write.mean = 120
T_MANIFEST_FILE.write.stddev = 15
```

**Configuration Tips:**
- **Fast storage (S3 Express, local SSD)**: mean=10-50ms, stddev=5-10ms
- **Standard cloud storage (S3)**: mean=50-150ms, stddev=20-40ms
- **Slower storage (cross-region)**: mean=200-500ms, stddev=50-100ms
- **min_latency**: Set to 1-2ms for very fast storage, 5-10ms for typical

### Workload Configuration

Model different transaction patterns:

```toml
[transaction]
# Transaction runtime (time to prepare data before commit)
runtime.min = 5000
runtime.mean = 10000
runtime.sigma = 1.5      # Lognormal shape parameter

# Table access patterns (Zipf distributions)
ntable.zipf = 2.0        # Tables per transaction (lower = more tables)
seltbl.zipf = 1.4        # Which tables selected (lower = more uniform)
seltblw.zipf = 1.2       # Tables written vs read (lower = more writes)
```

**Workload Scenarios:**
- **Hotspot workload**: High zipf values (2.0+) = few hot tables
- **Uniform workload**: Low zipf values (1.0-1.3) = even distribution
- **Read-heavy**: High seltblw.zipf = most txns write few tables
- **Write-heavy**: Low seltblw.zipf = most txns write many tables

## Interpret Results

### Success Rate vs Load

```
100ms inter-arrival: 85% success  → High contention, many conflicts
1000ms inter-arrival: 95% success → Moderate load, some conflicts
10000ms inter-arrival: 99% success → Low load, rare conflicts
```

**Guidelines:**
- **>95% success**: System under-loaded, room to increase throughput
- **90-95% success**: Healthy load with some contention
- **80-90% success**: High contention, consider scaling
- **<80% success**: Overloaded, retries not sufficient

### Commit Latency CDF

Understand the latency distribution:
- **Steep initial rise**: Most transactions succeed on first try
- **Steps in the curve**: Each step represents an additional retry
- **Long tail**: Transactions that fall many snapshots behind

**Example interpretation:**
```
P50 = 150ms   → Typical fast-path commit
P95 = 450ms   → Some retries needed (2-3 snapshots behind)
P99 = 1200ms  → Heavy conflicts (5+ snapshots behind)
```

### Retry Patterns

```
Mean retries: 0.5 → Mostly successful first try
Mean retries: 2.0 → Moderate conflicts, several snapshots to catch up
Mean retries: 5.0 → Heavy conflicts, reading many manifest lists
```

Higher retries mean:
- More manifest lists to read (n lists for n snapshots behind)
- More merge operations
- Higher latency and cost

## Run Tests

Verify simulator behavior with comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_simulator.py -v           # Determinism and parameter effects
pytest tests/test_conflict_resolution.py -v # New conflict resolution features

# Run specific test
pytest tests/test_conflict_resolution.py::TestMinimumLatency -v
```

**Test categories:**
- **Determinism**: Same seed = identical results
- **Parameter effects**: Latency, load, retries behave as expected
- **Conflict resolution**: CAS failures trigger manifest list reads
- **Storage latencies**: Minimum enforcement, distributions, read/write differences
- **Parallelism**: max_parallel limits are respected

**Example test output:**
```
✓ Minimum latency test passed
  Generated 1000 samples with mean=20, stddev=15
  Minimum value: 10.00ms (>= 10.0ms)

✓ High contention test passed
  Transactions: 145
  Retry rate: 68.3%
  Max retries: 7
  Mean retries: 3.12

✓ Commit latency increase test passed
  No retries: 142.35ms avg commit latency
  With retries (>=2): 523.18ms avg commit latency
  Increase: 380.83ms (due to manifest list reads)
```

## Understanding Conflict Resolution

When a transaction's CAS fails because catalog moved from s_k to s_{k+n}:

1. **Calculate snapshots behind**: `n = current_seq - txn_seq`
2. **Read n manifest lists**: One for each intermediate snapshot
3. **Batch with parallelism**: Read at most `max_parallel` in parallel
4. **Merge conflicts**: For each affected table, read and merge manifests
5. **Retry commit**: With updated snapshot version

**Example log (verbose mode):**
```
TXN 42 CAS Fail - 5 snapshots behind
TXN 42 Reading 5 manifest lists (max_parallel=4)
TXN 42 Read batch of 4 manifest lists (102.3ms)
TXN 42 Read batch of 1 manifest lists (97.8ms)
TXN 42 Merging table 3
TXN 42 Merging table 7
TXN 42 Retry commit
```

**Key insight:** The cost of conflict resolution scales with how far behind you are:
- 1 snapshot behind: Read 1 manifest list
- 5 snapshots behind: Read 5 manifest lists (batched by parallelism)
- 10 snapshots behind: Read 10 manifest lists (potentially very expensive)

This is why retry latency increases non-linearly with contention.

## Advanced Usage

### Custom Arrival Patterns

Test different client behavior:

```toml
# Fixed rate (deterministic)
inter_arrival.distribution = "fixed"
inter_arrival.value = 5000.0

# Bursty workload (uniform)
inter_arrival.distribution = "uniform"
inter_arrival.min = 1000.0
inter_arrival.max = 10000.0

# Variable rate (normal)
inter_arrival.distribution = "normal"
inter_arrival.mean = 5000.0
inter_arrival.std_dev = 1000.0
```

### Large-Scale Simulations

For production-like analysis:

```toml
[simulation]
duration_ms = 1000000000  # 1000 seconds (16.7 minutes)

[catalog]
num_tables = 100          # Realistic table count

[transaction]
runtime.mean = 30000      # 30 second transactions (realistic ETL)
```

**Note:** Larger simulations take longer but provide more stable statistics.

### Analyzing Results Programmatically

```python
import pandas as pd

# Load results
df = pd.read_parquet('results.parquet')

# Filter committed transactions
committed = df[df['status'] == 'committed']

# Analyze retry patterns
print(f"Transactions with 0 retries: {(committed['n_retries']==0).sum()}")
print(f"Transactions with 5+ retries: {(committed['n_retries']>=5).sum()}")

# Latency percentiles
print(committed['commit_latency'].quantile([0.5, 0.9, 0.95, 0.99]))

# Throughput over time
committed['time_bucket'] = committed['t_submit'] // 10000  # 10-second buckets
throughput = committed.groupby('time_bucket').size()
print(f"Mean throughput: {throughput.mean():.2f} txn/10sec")
```

## Troubleshooting

**High abort rate:**
- Increase `retry` limit
- Decrease load (increase `inter_arrival.scale`)
- Increase `num_tables` to reduce contention

**Unrealistic latencies:**
- Check `min_latency` is set appropriately (5-10ms typical)
- Verify `stddev` is reasonable (10-20% of mean)
- Ensure mean latencies match your storage system

**Slow simulation:**
- Reduce `duration_ms` for testing
- Increase `inter_arrival.scale` (fewer transactions)
- Use `--no-progress` flag

**Tests failing:**
- Check you activated virtual environment: `source bin/activate`
- Verify all dependencies installed: `pip install -r requirements.txt`
- Check config format matches current schema (see cfg.toml)

## Next Steps

1. **Explore parameter space**: Run sweep-combined with wide ranges
2. **Model your storage**: Configure latencies based on real measurements
3. **Test scaling strategies**: Vary num_tables, parallelism, retry limits
4. **Compare storage systems**: S3 vs S3 Express vs local using different latencies
5. **Analyze results**: Use summary.csv and Parquet files for detailed analysis
6. **Read the paper**: Understanding Iceberg's optimistic concurrency model
