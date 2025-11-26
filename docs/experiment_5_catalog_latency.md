# Experiment 5: Catalog Latency Impact

## Research Question
**How does catalog latency affect transaction throughput and the saturation point?**

All previous experiments (2.x, 3.x) used "infinitely fast" catalog operations (T_CAS = 1ms, T_METADATA_ROOT = 1ms) to isolate conflict resolution costs. This experiment measures the impact of realistic catalog latencies.

## Motivation
Real-world catalog implementations have varying latencies:
- **15ms**: S3 Express One Zone (single-AZ, fastest)
- **50ms**: Standard S3 in same region (typical)
- **100ms**: S3 cross-region or slower operations
- **200ms**: Slower cloud storage / busy systems
- **500ms**: GCP standard latency
- **1000ms**: 2x GCP (slowest realistic case)

The CAS operation is the critical path for every transaction commit. Understanding how catalog latency affects throughput is essential for:
1. Choosing catalog backends (Iceberg REST, Polaris, Unity Catalog, etc.)
2. Sizing catalog infrastructure
3. Understanding throughput ceilings in production

## Experiment Design

### Parameters

#### Catalog Latencies (sweep these together)
- `T_CAS.mean`: [15, 50, 100, 200, 500, 1000] ms
- `T_METADATA_ROOT.{read,write}.mean`: Same as T_CAS (catalog operations)
- Stddev: 10% of mean (realistic variation)

#### Storage Latencies (fixed, realistic S3)
- `T_MANIFEST_LIST`: 50ms read, 60ms write
- `T_MANIFEST_FILE`: 50ms read, 60ms write
- `MIN_LATENCY`: 1ms

#### Load Sweep (to find saturation)
- `inter_arrival.scale`: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms
- Maps to ~100, 50, 20, 10, 5, 2, 1, 0.5, 0.2 txn/sec

#### Seeds
- 5 seeds per configuration (for statistical stability)

### Sub-Experiments

#### Exp 5.1: Single Table, False Conflicts
**Goal**: Isolate pure catalog latency impact (no multi-table or real conflict overhead)

Parameters:
- Catalog latency: [15, 50, 100, 200, 500, 1000] ms
- Load sweep: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms
- num_tables: 1
- real_conflict_probability: 0.0
- Seeds: 5

Total runs: 6 latencies × 9 loads × 5 seeds = **270 runs**

Expected result: Throughput ceiling inversely proportional to catalog latency
- 15ms catalog → ~66 txn/sec ceiling
- 50ms catalog → ~20 txn/sec ceiling
- 1000ms catalog → ~1 txn/sec ceiling

#### Exp 5.2: Multi-Table, False Conflicts
**Goal**: Understand catalog latency impact with multi-table contention

Parameters:
- Catalog latency: [15, 50, 100, 200, 500, 1000] ms
- Load sweep: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms
- num_tables: [1, 5, 20, 50]
- real_conflict_probability: 0.0
- Seeds: 5

Total runs: 6 latencies × 9 loads × 4 tables × 5 seeds = **1,080 runs**

Expected result: With slow catalog (500-1000ms):
- Hot table bottleneck may matter less (catalog is slower than contention)
- More tables might help more than in fast-catalog case

#### Exp 5.3: Transaction Partitioning with Catalog Latency
**Goal**: How does transaction partitioning (num_groups) interact with catalog latency?

Parameters:
- Catalog latency: [15, 50, 100, 200, 500, 1000] ms
- Load sweep: [10, 20, 50, 100, 200, 500, 1000, 2000, 5000] ms
- num_tables: 20 (fixed)
- num_groups: [1, 2, 5, 10, 20] (from catalog-level to table-level conflicts)
- Seeds: 5

Total runs: 6 latencies × 9 loads × 5 groups × 5 seeds = **1,350 runs**

Expected result: Transaction partitioning effectiveness depends on catalog speed
- With fast catalog (15ms): More groups significantly reduces contention
- With slow catalog (1000ms): Catalog latency dominates, grouping matters less
- Crossover point: Where catalog speed overtakes contention as bottleneck

**Key insight**: Real conflicts primarily extend conflict window - if catalog is already slow, this effect is less significant. Transaction partitioning directly affects contention, which is more relevant when studying catalog bottlenecks.

Total runs across all 5.x experiments: **2,700 runs**
With 96 parallel jobs at ~1 hour per run: ~28 hours wall-clock time

## Implementation Requirements

### 1. Simulator Modifications: NONE REQUIRED ✓
The simulator already supports:
- `T_CAS` configuration (used in icecap/main.py:847)
- `T_METADATA_ROOT` configuration (used in icecap/main.py:705, 730)
- Both use normal distributions with mean/stddev

### 2. Configuration File Creation
Create base configs:
- `experiment_configs/exp5_1_single_table_catalog_latency.toml`
- `experiment_configs/exp5_2_multi_table_catalog_latency.toml`
- `experiment_configs/exp5_3_transaction_partitioning_catalog_latency.toml`

### 3. Experiment Runner Updates
Update `scripts/run_baseline_experiments.sh` to support:
- `--exp5.1`, `--exp5.2`, `--exp5.3` flags
- Sweeping over catalog latencies and num_groups in addition to loads

### 4. Analysis Scripts
Modify `icecap/saturation_analysis.py` or create new analysis:
- Plot throughput vs load, grouped by catalog latency
- Show how saturation point shifts with catalog latency
- Compare catalog latency impact across different workload types

## Key Insights Expected

1. **Catalog as Bottleneck**: With slow catalog (500-1000ms), catalog latency dominates throughput ceiling

2. **Crossover Point**: Find the catalog latency where it becomes the bottleneck vs other factors (hot table contention, transaction partitioning)

3. **Catalog Backend Selection**: Quantify benefit of fast catalog (S3 Express vs standard S3 vs GCP)

4. **Partitioning Effectiveness**: At what catalog latency does transaction partitioning stop helping?
   - Fast catalog: Partitioning reduces contention significantly
   - Slow catalog: Catalog itself is bottleneck, partitioning helps less

5. **Concurrency Scaling**: How does catalog latency affect the benefit of concurrent transactions?

## Next Steps

1. Create configuration files for each sub-experiment
2. Update experiment runner to support exp5.x
3. Run experiments (38 hours with 96 parallel jobs)
4. Generate plots showing throughput vs catalog latency
5. Document findings in paper
