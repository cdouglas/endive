# Icecap: Iceberg Catalog Simulator

A discrete-event simulator for exploring commit latency tradeoffs in shared-storage catalog formats like Apache Iceberg. The simulator models the optimistic concurrency control (OCC) protocol with compare-and-swap (CAS) operations, conflict resolution, and retry logic.

## Overview

When multiple writers attempt to commit changes to an Iceberg table simultaneously, conflicts can occur. A failed pointer swap at the catalog requires additional round trips to:

1. Read the manifest file (JSON)
2. Merge the old snapshot and write a new manifest file
3. Read the manifest list
4. Merge the updated manifest list with changes in this transaction
5. For all conflicts in manifest files: merge and rewrite manifest files
6. Retry the pointer swap at the catalog

This simulator helps explore how different parameters (number of concurrent clients, catalog latency, table distribution) affect end-to-end commit latency and success rates.

## Installation

```bash
# Create virtual environment
python3 -m venv .

# Activate virtual environment
source bin/activate  # Linux/Mac
# or
.\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package (for CLI tools)
pip install -e .
```

## Quick Start

### 1. Run a single simulation

```bash
# Use default configuration (cfg.toml)
python -m icecap.main

# Use custom configuration
python -m icecap.main my_config.toml

# Verbose logging
python -m icecap.main -v

# Quiet mode (errors only)
python -m icecap.main -q

# Skip confirmation prompt (for automation)
python -m icecap.main -y

# Disable progress bar
python -m icecap.main --no-progress
```

The simulator will:
1. Display configuration summary
2. Ask for confirmation (unless `-y` or `-q` is used)
3. Show progress bar during simulation (unless `--no-progress`, `-v`, or `-q` is used)
4. Export results and display summary statistics

### 2. Run experiments (parameter sweeps)

```bash
# Sweep over different client loads (inter-arrival times)
python -m icecap.experiment sweep-clients \
    --times 100 500 1000 2000 5000 10000

# Sweep over different catalog latencies
python -m icecap.experiment sweep-latency \
    --latencies 10 50 100 200 500 1000

# Combined sweep (both parameters)
python -m icecap.experiment sweep-combined \
    --times 500 1000 5000 \
    --latencies 50 100 200

# Dry run (generate configs without running)
python -m icecap.experiment sweep-clients --dry-run

# Specify custom base config and output directory
python -m icecap.experiment sweep-clients \
    -b my_base_config.toml \
    -o my_experiments
```

### 3. Analyze results and generate plots

```bash
# Generate all plots and summary table
python -m icecap.analysis all -i experiments -o plots

# Generate specific plots
python -m icecap.analysis cdf -i experiments -o plots
python -m icecap.analysis success-rate -i experiments -o plots
python -m icecap.analysis latency-impact -i experiments -o plots

# Analyze specific parameter
python -m icecap.analysis cdf --param cas_latency
python -m icecap.analysis success-rate --param inter_arrival
```

## Configuration

The simulator uses TOML configuration files. Here's an overview of the parameters:

### Simulation Parameters

```toml
[simulation]
duration_ms = 100000000      # Simulation duration (100 seconds)
output_path = "results.parquet"  # Output file path
seed = null                  # Random seed (null = random)
```

### Catalog Configuration

```toml
[catalog]
num_tables = 10              # Number of tables in catalog

# Table grouping (optional, default: num_groups = 1)
num_groups = 1               # Number of table groups
group_size_distribution = "uniform"  # "uniform" or "longtail"

# Longtail distribution parameters
longtail.large_group_fraction = 0.5
longtail.medium_groups_count = 3
longtail.medium_group_fraction = 0.3
```

**Table Grouping:**

Tables can be partitioned into groups to model different conflict detection granularities:

- **`num_groups = 1`** (default): Catalog-level conflicts - any concurrent writes conflict
- **`num_groups = T`** (where T = num_tables): Table-level conflicts - transactions only conflict if they touch the same tables
- **`1 < num_groups < T`**: Group-level isolation - useful for modeling multi-tenant scenarios

Transactions never span group boundaries. The simulator enforces this by selecting all transaction tables from a single random group.

**Distribution types:**
- **`uniform`**: All groups have approximately equal size (T/G tables each)
- **`longtail`**: One large group, a few medium groups, and many small groups (models skewed workloads)

### Transaction Parameters

```toml
[transaction]
retry = 10                   # Maximum number of retries

# Transaction runtime (lognormal distribution)
runtime.min = 5000           # Minimum runtime (ms)
runtime.mean = 10000         # Mean runtime (ms)
runtime.sigma = 1.5          # Sigma for lognormal distribution

# Inter-arrival time distribution
inter_arrival.distribution = "exponential"  # Options: fixed, exponential, uniform, normal
inter_arrival.scale = 5000.0                # For exponential (mean inter-arrival time)
inter_arrival.value = 5000.0                # For fixed distribution
inter_arrival.min = 1000.0                  # For uniform distribution
inter_arrival.max = 10000.0                 # For uniform distribution
inter_arrival.mean = 5000.0                 # For normal distribution
inter_arrival.std_dev = 1000.0              # For normal distribution

# Table selection (Zipf distributions)
ntable.zipf = 2.0            # Number of tables per transaction
seltbl.zipf = 1.4            # Which tables are selected (0 most likely)
seltblw.zipf = 1.2           # Number of tables written (subset of read)
```

### Storage Latencies

Storage operations use normal distributions with mean and standard deviation:

```toml
[storage]
# Maximum parallel manifest operations during conflict resolution
max_parallel = 4

# Minimum latency for any storage operation (ms) - prevents unrealistic zeros
min_latency = 5

# CAS operation latency (ms) - normal distribution
T_CAS.mean = 100
T_CAS.stddev = 10

# Metadata root read/write latency (ms)
T_METADATA_ROOT.read.mean = 100
T_METADATA_ROOT.read.stddev = 10
T_METADATA_ROOT.write.mean = 120
T_METADATA_ROOT.write.stddev = 15

# Manifest list read/write latency (ms)
T_MANIFEST_LIST.read.mean = 100
T_MANIFEST_LIST.read.stddev = 10
T_MANIFEST_LIST.write.mean = 120
T_MANIFEST_LIST.write.stddev = 15

# Manifest file read/write latency (ms)
T_MANIFEST_FILE.read.mean = 100
T_MANIFEST_FILE.read.stddev = 10
T_MANIFEST_FILE.write.mean = 120
T_MANIFEST_FILE.write.stddev = 15
```

**Conflict Resolution:**
When a CAS fails because the catalog has moved from snapshot s_k to s_{k+n}, the simulator:
1. Reads n manifest lists (one for each intermediate snapshot)
2. Processes at most `max_parallel` manifest lists in parallel
3. Merges conflicts for each affected table
4. Retries the commit with updated snapshot version

## Understanding Inter-Arrival Time

**Inter-arrival time controls the effective number of concurrent clients:**

- **Lower inter-arrival time** = More concurrent clients = Higher contention
  - Example: `inter_arrival.scale = 100` means ~10 transactions/second with high concurrency
- **Higher inter-arrival time** = Fewer concurrent clients = Lower contention
  - Example: `inter_arrival.scale = 10000` means ~0.1 transactions/second with low concurrency

The `distribution` parameter controls the variability:
- `"fixed"`: Transactions arrive at exact intervals (deterministic)
- `"exponential"`: Poisson arrival process (most realistic for independent clients)
- `"uniform"`: Uniformly distributed inter-arrival times
- `"normal"`: Normally distributed inter-arrival times

## Output Format

Results are exported as Parquet files containing:

| Column | Type | Description |
|--------|------|-------------|
| `txn_id` | int | Transaction ID |
| `t_submit` | int | Submission time (ms) |
| `t_runtime` | int | Transaction runtime (ms) |
| `t_commit` | int | Commit time (ms) or -1 if aborted |
| `commit_latency` | int | Time from ready to commit (ms) |
| `total_latency` | int | Total time from submit to commit/abort |
| `n_retries` | int | Number of retry attempts |
| `n_tables_read` | int | Number of tables read |
| `n_tables_written` | int | Number of tables written |
| `status` | str | "committed" or "aborted" |

## Generated Plots

The analysis tool generates several plots:

### 1. CDF of Commit Latency
Shows the cumulative distribution of commit latencies for different client loads or catalog latencies.

**Interpretation:**
- Steeper curves = more consistent latency
- Longer tails = occasional high latencies
- Compare curves to see impact of parameter changes

### 2. Success Rate vs Client Load
Shows how success rate and throughput change with different inter-arrival times.

**Interpretation:**
- Higher inter-arrival time (fewer clients) = higher success rate
- Lower inter-arrival time (more clients) = more contention, more aborts
- Throughput peaks at optimal concurrency level

### 3. Catalog Latency Impact
Multi-panel plot showing how catalog CAS latency affects:
- Mean commit latency
- P95 commit latency
- Retry rate
- Success rate

**Interpretation:**
- Higher CAS latency amplifies the cost of retries
- Different inter-arrival times show how concurrency interacts with latency
- Can identify optimal operating points

### 4. Summary Table (CSV)
Contains aggregated statistics for all experiments.

## Table Grouping Use Cases

### Catalog-Level vs Table-Level Conflicts

Compare how different catalog designs affect performance:

```bash
# Catalog-level conflicts (like Iceberg v1)
python -m icecap.main cfg.toml  # with num_groups = 1

# Table-level conflicts (like Iceberg v2 with independent table commits)
# Edit cfg.toml: set num_groups = num_tables
python -m icecap.main cfg.toml
```

With catalog-level conflicts, any two concurrent writes conflict. With table-level conflicts, only transactions touching the same tables conflict, reducing contention.

### Multi-Tenant Workloads

Model isolation between tenants using groups:

```toml
[catalog]
num_tables = 100
num_groups = 10              # 10 tenants
group_size_distribution = "longtail"
longtail.large_group_fraction = 0.4   # One large tenant
longtail.medium_groups_count = 3       # Three medium tenants
longtail.medium_group_fraction = 0.3   # Remainder split among small tenants
```

This models a realistic scenario where one large tenant dominates, a few medium tenants share resources, and many small tenants have minimal activity.

### Hotspot Analysis

Examine how table popularity affects contention:

```toml
[catalog]
num_tables = 50
num_groups = 5
group_size_distribution = "uniform"

[transaction]
seltbl.zipf = 2.0   # High skew - group 0 is "hot"
```

Since table selection is Zipf-distributed and transactions are confined to groups, group 0 (containing tables 0-9) will experience the highest contention.

## Example Workflow

```bash
# 1. Edit base configuration
vim cfg.toml

# 2. Run experiments sweeping over client loads
python -m icecap.experiment sweep-clients \
    --times 100 250 500 1000 2500 5000 10000 \
    --dist exponential \
    -o experiments/clients

# 3. Run experiments sweeping over catalog latencies
python -m icecap.experiment sweep-latency \
    --latencies 10 25 50 100 250 500 1000 \
    -o experiments/latency

# 4. Run combined experiment for detailed analysis
python -m icecap.experiment sweep-combined \
    --times 500 1000 5000 \
    --latencies 50 100 200 \
    -o experiments/combined

# 5. Generate plots
python -m icecap.analysis all -i experiments/clients -o plots/clients
python -m icecap.analysis all -i experiments/latency -o plots/latency
python -m icecap.analysis all -i experiments/combined -o plots/combined

# 6. View results
open plots/clients/*.png
open plots/clients/summary.csv
```

## Architecture

### Core Components

- **`icecap/main.py`**: Simulator implementation using SimPy
  - `Catalog`: Versioned catalog with CAS operations
  - `Txn`: Transaction dataclass with metadata
  - `txn_gen()`: Transaction generator
  - `txn_commit()`: Commit logic with retry and conflict resolution

- **`icecap/capstats.py`**: Statistics collection
  - `Stats`: Collects transaction metrics and exports to Parquet
  - Utility functions for distributions

- **`icecap/experiment.py`**: Experiment runner
  - Generates configuration files for parameter sweeps
  - Runs experiments in batch

- **`icecap/analysis.py`**: Analysis and plotting
  - Loads Parquet results
  - Generates CDF plots, success rate plots, latency impact analysis
  - Exports summary tables

## Markov Model Alternative

The retry process can be modeled analytically as a Discrete-Time Markov Chain (DTMC):

- **States**: {0, 1, 2, ..., N} representing retry attempts
- **Transitions**:
  - Success with probability `p`: transition to absorbing "committed" state
  - Failure with probability `(1-p)`: transition to next retry state
  - At state N: transition to absorbing "aborted" state

The expected commit time can be calculated analytically:

```
E[T] = RTT * (1 + (1-p) * sum_{k=1}^{N} (1-p)^k * delay(k))
```

Where `delay(k)` is the backoff delay for retry k.

This analytical model is implemented in `rtt.py` and `plot_rtt.py` for validation and quick parameter exploration without running full simulations.

## Testing

The simulator includes comprehensive tests to verify correctness and parameter effects.

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test classes
pytest tests/test_simulator.py::TestDeterminism -v
pytest tests/test_simulator.py::TestParameterEffects -v
```

### Test Coverage

**Table Grouping Tests** (`test_table_groups.py`):
- Uniform and longtail group size distributions
- Transactions stay within single group
- Table-level conflict detection when num_groups = num_tables
- Warning emission when transaction exceeds group size
- Deterministic group partitioning with seed

**Determinism Tests** (`test_simulator.py`):
- Verifies identical results with same random seed
- Confirms different seeds produce different outcomes

**Parameter Effect Tests** (`test_simulator.py`):
- Lower inter-arrival time increases contention and retries
- Higher CAS latency increases commit times
- More retries improve success rates
- Fewer tables increase contention (more overlapping transactions)

**Conflict Resolution Tests** (`test_conflict_resolution.py`):
- High contention causes CAS failures and retries
- Commit latency increases with number of retries
- Different parallelism limits affect performance under contention

**Storage Latency Tests** (`test_conflict_resolution.py`):
- Minimum latency is enforced (prevents unrealistic zeros)
- Latencies follow approximately normal distribution
- Read and write latencies are differentiated
- Stochastic latencies work correctly in simulation

These tests serve as both validation and documentation of expected simulator behavior.

## License

This is a research prototype. Use at your own risk.

## Contributing

This is a personal research project. Feel free to fork and experiment!
