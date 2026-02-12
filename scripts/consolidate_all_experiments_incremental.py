#!/usr/bin/env python3
"""
Memory-efficient consolidation using incremental PyArrow writing.

This version processes experiments in batches and appends to the Parquet file
incrementally, avoiding memory exhaustion.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tomli
from tqdm import tqdm


def flatten_config(config: dict, prefix: str = '') -> List[Tuple[str, str]]:
    """Flatten nested TOML config to list of (key, value) pairs."""
    items = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(flatten_config(value, full_key))
        else:
            items.append((full_key, str(value)))
    return items


def load_and_normalize_schema(parquet_path: str) -> pd.DataFrame:
    """Load results.parquet with automatic schema normalization."""
    df = pd.read_parquet(parquet_path)

    # Normalize time/latency columns: float64 → int64
    time_cols = ['t_commit', 'commit_latency', 'total_latency']
    for col in time_cols:
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].round().astype('int64')

    # Normalize count columns: int64 → int8
    count_cols = ['n_retries', 'n_tables_read', 'n_tables_written']
    for col in count_cols:
        if col in df.columns and df[col].dtype == 'int64':
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')

    return df


def consolidate_incremental(
    base_dir: str = 'experiments',
    output_path: str = 'experiments/consolidated.parquet',
    batch_size: int = 50,
    compression: str = 'zstd',
    compression_level: int = 3
):
    """
    Consolidate experiments incrementally to avoid memory exhaustion.

    Processes experiments in batches and writes to temporary sorted files,
    then merges them into final consolidated file.
    """
    print("=" * 80)
    print("INCREMENTAL EXPERIMENT CONSOLIDATION")
    print("=" * 80)
    print(f"\nBase directory: {base_dir}")
    print(f"Output file: {output_path}")
    print(f"Batch size: {batch_size} experiments")
    print(f"Compression: {compression} (level {compression_level})")

    # Find all experiments
    print("\n" + "-" * 80)
    print("PHASE 1: Scanning experiment directories")
    print("-" * 80)

    experiments = []
    # Match directories with hash suffix (e.g., baseline_s3-abc123, exp2_1-def456)
    for exp_dir in sorted(Path(base_dir).glob('*-*')):
        if not exp_dir.is_dir():
            continue

        dir_name = exp_dir.name
        # Split on last hyphen to separate name from hash
        exp_name, exp_hash = dir_name.rsplit('-', 1)
        # Skip if hash doesn't look like a hex string (at least 6 hex chars)
        if not (len(exp_hash) >= 6 and all(c in '0123456789abcdef' for c in exp_hash)):
            continue

        seed_dirs = [d for d in exp_dir.iterdir()
                     if d.is_dir() and (d / 'results.parquet').exists()]

        if seed_dirs:
            experiments.append((exp_name, exp_hash, exp_dir, seed_dirs))

    print(f"\nFound {len(experiments)} experiments")
    total_seeds = sum(len(seeds) for _, _, _, seeds in experiments)
    print(f"Total seeds: {total_seeds}")

    # Define schema
    schema = pa.schema([
        ('txn_id', pa.int64()),
        ('t_submit', pa.int64()),
        ('t_runtime', pa.int64()),
        ('t_commit', pa.int64()),
        ('commit_latency', pa.int64()),
        ('total_latency', pa.int64()),
        ('n_retries', pa.int8()),
        ('n_tables_read', pa.int8()),
        ('n_tables_written', pa.int8()),
        ('status', pa.string()),
        ('exp_name', pa.string()),
        ('exp_hash', pa.string()),
        ('seed', pa.int64()),
        ('config', pa.map_(pa.string(), pa.string()))
    ])

    # Process experiments in batches and write incrementally
    print("\n" + "-" * 80)
    print("PHASE 2: Processing and writing incrementally")
    print("-" * 80)

    writer = None
    total_rows = 0

    try:
        for i, (exp_name, exp_hash, exp_dir, seed_dirs) in enumerate(tqdm(experiments, desc="Processing")):
            # Load config
            cfg_path = exp_dir / 'cfg.toml'
            if cfg_path.exists():
                try:
                    with open(cfg_path, 'rb') as f:
                        config = tomli.load(f)
                    config_map = flatten_config(config)
                except Exception as e:
                    config_map = [('error', f'Failed to load: {e}')]
            else:
                config_map = [('warning', 'No config file')]

            # Process each seed
            for seed_dir in seed_dirs:
                parquet_path = seed_dir / 'results.parquet'
                if not parquet_path.exists():
                    continue

                try:
                    # Load seed data
                    df = load_and_normalize_schema(str(parquet_path))

                    # Add metadata
                    df['exp_name'] = exp_name
                    df['exp_hash'] = exp_hash
                    df['seed'] = int(seed_dir.name)
                    df['config'] = [config_map] * len(df)

                    # Sort by t_submit only
                    # Directory traversal order already provides exp_name, exp_hash, seed ordering
                    df = df.sort_values('t_submit')

                    # Convert to PyArrow table
                    table = pa.Table.from_pandas(df, schema=schema)

                    # Write to file (append mode)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path,
                            schema,
                            compression=compression,
                            compression_level=compression_level if compression == 'zstd' else None,
                            use_dictionary=['exp_name', 'exp_hash', 'status'],
                            write_statistics=True,
                            version='2.6'
                        )

                    writer.write_table(table)
                    total_rows += len(df)

                    # Clear memory
                    del df, table

                except Exception as e:
                    print(f"\n  ERROR processing {parquet_path}: {e}")
                    continue

        # Close writer
        if writer is not None:
            writer.close()

        # Print summary
        file_size = Path(output_path).stat().st_size
        print("\n" + "=" * 80)
        print("CONSOLIDATION COMPLETE")
        print("=" * 80)
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size / (1024**2):.1f} MB ({file_size / (1024**3):.2f} GB)")
        print(f"Total rows: {total_rows:,}")
        print(f"Experiments: {len(experiments)}")
        print(f"Seeds: {total_seeds}")
        print(f"\n✓ Original files PRESERVED (not deleted)")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient consolidation')
    parser.add_argument('--base-dir', default='experiments')
    parser.add_argument('--output', default='experiments/consolidated.parquet')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--compression', default='zstd', choices=['zstd', 'snappy'])
    parser.add_argument('--compression-level', type=int, default=3)

    args = parser.parse_args()

    success = consolidate_incremental(
        base_dir=args.base_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        compression=args.compression,
        compression_level=args.compression_level
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
