# dump_results.py - Results.parquet Analysis Tool

## Schema

The `results.parquet` file contains transaction execution records with the following schema:

| Column           | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| txn_id           | int64   | Transaction ID                                 |
| t_submit         | int64   | Submission time (ms from simulation start)     |
| t_runtime        | int64   | Transaction runtime (ms)                       |
| t_commit         | int64   | Commit completion time (ms)                    |
| commit_latency   | int64   | Time spent in commit protocol (ms)             |
| total_latency    | int64   | Total transaction latency (ms)                 |
| n_retries        | int8    | Number of commit retries                       |
| n_tables_read    | int8    | Number of tables read                          |
| n_tables_written | int8    | Number of tables written                       |
| status           | object  | Transaction status ('committed' or 'aborted')  |

## Usage

### Basic Usage

Show schema and summary:
```bash
python scripts/dump_results.py results.parquet
```

### Display Options

Show first N rows:
```bash
python scripts/dump_results.py results.parquet --head 20
```

Show last N rows:
```bash
python scripts/dump_results.py results.parquet --tail 10
```

Show all rows:
```bash
python scripts/dump_results.py results.parquet --all
```

Skip schema/summary:
```bash
python scripts/dump_results.py results.parquet --head 10 --no-schema --no-summary
```

### Filtering

Filter by status:
```bash
python scripts/dump_results.py results.parquet --status committed
python scripts/dump_results.py results.parquet --status aborted
```

Filter by submission time (steady-state window):
```bash
# Show transactions submitted between 15-45 minutes (900s - 2700s)
python scripts/dump_results.py results.parquet --time-range 900000 2700000
```

Filter by minimum retries:
```bash
# Show transactions with at least 3 retries
python scripts/dump_results.py results.parquet --retries-min 3
```

### Column Selection

Show specific columns only:
```bash
python scripts/dump_results.py results.parquet --head 10 \
  --columns txn_id t_submit status commit_latency n_retries
```

Sort by column:
```bash
python scripts/dump_results.py results.parquet --head 10 --sort commit_latency
```

### Output Formats

CSV format:
```bash
python scripts/dump_results.py results.parquet --all --format csv > output.csv
```

JSON format:
```bash
python scripts/dump_results.py results.parquet --all --format json > output.json
```

### Statistics

Show detailed statistics by status:
```bash
python scripts/dump_results.py results.parquet --stats
```

## Examples

### Analyze steady-state window

```bash
# Show statistics for transactions in steady-state (15-45 min)
python scripts/dump_results.py \
  experiments/exp2_1_*/12345/results.parquet \
  --time-range 900000 2700000 \
  --stats
```

### Find high-retry transactions

```bash
# List transactions with 5+ retries
python scripts/dump_results.py results.parquet \
  --retries-min 5 \
  --columns txn_id t_submit n_retries commit_latency status \
  --sort n_retries
```

### Compare committed vs aborted

```bash
# Committed statistics
python scripts/dump_results.py results.parquet --status committed --stats

# Aborted statistics
python scripts/dump_results.py results.parquet --status aborted --stats
```

### Export filtered data

```bash
# Export successful transactions from steady-state window as CSV
python scripts/dump_results.py results.parquet \
  --status committed \
  --time-range 900000 2700000 \
  --format csv \
  --no-schema --no-summary \
  > steady_state_committed.csv
```

### Quick inspection

```bash
# Quick look at first 5 transactions
python scripts/dump_results.py results.parquet --head 5 --no-schema --no-summary
```

## Tips

1. **Time ranges**: Simulation time starts at 0. For 1-hour simulations with 15-minute warmup/cooldown, steady-state is 900000-2700000ms.

2. **Combining filters**: You can combine multiple filters:
   ```bash
   python scripts/dump_results.py results.parquet \
     --status committed \
     --time-range 900000 2700000 \
     --retries-min 3
   ```

3. **Pipeline with other tools**: Use `--format csv` with standard Unix tools:
   ```bash
   python scripts/dump_results.py results.parquet --all --format csv | \
     cut -d',' -f1,5,7 | head -20
   ```

4. **Help**: See all options:
   ```bash
   python scripts/dump_results.py --help
   ```
