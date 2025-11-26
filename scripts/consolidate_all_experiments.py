#!/usr/bin/env python3
"""
DEPRECATED: This is the old in-memory consolidation script that may cause OOM.

Use the incremental version instead:
    python scripts/consolidate_all_experiments_incremental.py

The incremental version:
- Uses streaming writes (no OOM)
- Processes experiments one at a time
- Same output format and features
- ~22 minutes to consolidate all experiments

This script is kept for reference only and will be removed in a future version.

---

Original description:
Consolidate all experiment results into experiments/consolidated.parquet.

IMPORTANT: Does NOT delete any original files. All original data is preserved.

Usage:
    # Dry run (show what would be done)
    python scripts/consolidate_all_experiments.py --dry-run

    # Live run
    python scripts/consolidate_all_experiments.py

    # Custom settings
    python scripts/consolidate_all_experiments.py --compression zstd --compression-level 9
"""

import sys

print("=" * 80)
print("DEPRECATED SCRIPT")
print("=" * 80)
print()
print("This script (consolidate_all_experiments.py) is deprecated.")
print("It may cause out-of-memory errors on large datasets.")
print()
print("Use the incremental version instead:")
print("  python scripts/consolidate_all_experiments_incremental.py")
print()
print("The incremental version has the same output format but uses")
print("streaming writes to avoid memory issues.")
print()
print("=" * 80)
sys.exit(1)

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tomli
from tqdm import tqdm


def find_all_experiments(base_dir: str) -> List[Tuple[str, str, Path, List[Path]]]:
    """
    Find all experiments with results.parquet files.

    Returns:
        List of (exp_name, exp_hash, exp_dir, [seed_dirs]) tuples
    """
    experiments = []

    for exp_dir in sorted(Path(base_dir).glob('exp*')):
        if not exp_dir.is_dir():
            continue

        # Parse experiment name and hash
        dir_name = exp_dir.name
        if '-' in dir_name:
            exp_name, exp_hash = dir_name.rsplit('-', 1)
        else:
            exp_name = dir_name
            exp_hash = 'unknown'

        # Find seed directories with results.parquet
        seed_dirs = []
        for seed_dir in sorted(exp_dir.iterdir()):
            if seed_dir.is_dir() and (seed_dir / 'results.parquet').exists():
                seed_dirs.append(seed_dir)

        if not seed_dirs:
            continue

        experiments.append((exp_name, exp_hash, exp_dir, seed_dirs))

    return experiments


def flatten_config(config: Dict, prefix: str = '') -> List[Tuple[str, str]]:
    """
    Flatten nested TOML config to list of (key, value) pairs.

    Example:
        {'transaction': {'retry': 10, 'inter_arrival': {'scale': 10}}}
        →
        [('transaction.retry', '10'),
         ('transaction.inter_arrival.scale', '10')]
    """
    items = []
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recurse into nested dict
            items.extend(flatten_config(value, full_key))
        else:
            # Convert value to string
            items.append((full_key, str(value)))

    return items


def load_and_normalize_schema(parquet_path: str) -> pd.DataFrame:
    """
    Load results.parquet with automatic schema normalization.

    Converts old schema (float64) → new schema (int64) losslessly.
    """
    df = pd.read_parquet(parquet_path)

    # Normalize time/latency columns: float64 → int64
    time_cols = ['t_commit', 'commit_latency', 'total_latency']
    for col in time_cols:
        if col in df.columns and df[col].dtype == 'float64':
            # Round to nearest int (should be exact for milliseconds)
            df[col] = df[col].round().astype('int64')

    # Normalize count columns: int64 → int8
    count_cols = ['n_retries', 'n_tables_read', 'n_tables_written']
    for col in count_cols:
        if col in df.columns and df[col].dtype == 'int64':
            # Verify values fit in int8 range [-128, 127]
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
            else:
                print(f"    WARNING: {col} values exceed int8 range in {parquet_path}")
                print(f"             min={df[col].min()}, max={df[col].max()}, keeping int64")

    return df


def consolidate_all_experiments(
    base_dir: str = 'experiments',
    output_path: str = 'experiments/consolidated.parquet',
    compression: str = 'zstd',
    compression_level: int = 3,
    dry_run: bool = False
):
    """
    Consolidate all experiments into single Parquet file.

    Args:
        base_dir: Directory containing experiment subdirectories
        output_path: Output consolidated file path
        compression: Compression codec ('zstd' recommended, 'snappy', 'gzip', 'brotli')
        compression_level: Compression level (1-22 for zstd, 3 recommended)
        dry_run: If True, show what would be done without writing
    """
    print("=" * 80)
    print("EXPERIMENT CONSOLIDATION")
    print("=" * 80)
    print(f"\nBase directory: {base_dir}")
    print(f"Output file: {output_path}")
    print(f"Compression: {compression} (level {compression_level})")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"\nIMPORTANT: Original files will be PRESERVED (not deleted)")

    # Find all experiments
    print("\n" + "-" * 80)
    print("PHASE 1: Scanning experiment directories")
    print("-" * 80)

    experiments = find_all_experiments(base_dir)

    if not experiments:
        print(f"ERROR: No experiments found in {base_dir}")
        return False

    # Print summary
    total_seeds = sum(len(seeds) for _, _, _, seeds in experiments)
    print(f"\nFound {len(experiments)} experiment directories")
    print(f"Total seeds: {total_seeds}")

    # Group by family
    families = {}
    for exp_name, exp_hash, exp_dir, seeds in experiments:
        # Extract family (everything before last underscore + number)
        family = exp_name
        families[family] = families.get(family, [])
        families[family].append((exp_hash, len(seeds)))

    print("\nExperiment families:")
    for family in sorted(families.keys()):
        exps = families[family]
        total_seeds_in_family = sum(s for _, s in exps)
        print(f"  {family:45s}: {len(exps):3d} experiments, {total_seeds_in_family:4d} seeds")

    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN COMPLETE")
        print("=" * 80)
        print(f"\nWould process {len(experiments)} experiments with {total_seeds} seeds")
        print("\nTo run consolidation, execute without --dry-run flag:")
        print(f"  python scripts/consolidate_all_experiments.py")
        return True

    # Process all experiments
    print("\n" + "-" * 80)
    print("PHASE 2: Loading and normalizing data")
    print("-" * 80)

    all_data = []
    processed_experiments = 0
    processed_seeds = 0
    total_rows = 0
    errors = []

    for exp_name, exp_hash, exp_dir, seed_dirs in tqdm(experiments, desc="Processing experiments"):
        # Load config
        cfg_path = exp_dir / 'cfg.toml'
        if cfg_path.exists():
            try:
                with open(cfg_path, 'rb') as f:
                    config = tomli.load(f)
                config_map = flatten_config(config)
            except Exception as e:
                print(f"\n  WARNING: Error reading cfg.toml for {exp_name}-{exp_hash}: {e}")
                config_map = [('error', f'Failed to load config: {e}')]
        else:
            print(f"\n  WARNING: No cfg.toml for {exp_name}-{exp_hash}")
            config_map = [('warning', 'No config file found')]

        # Process each seed
        for seed_dir in seed_dirs:
            seed = seed_dir.name
            parquet_path = seed_dir / 'results.parquet'

            if not parquet_path.exists():
                print(f"\n  WARNING: No results.parquet in {seed_dir}")
                continue

            try:
                # Load and normalize schema
                df = load_and_normalize_schema(str(parquet_path))

                # Add metadata columns
                df['exp_name'] = exp_name
                df['exp_hash'] = exp_hash
                df['seed'] = int(seed)
                df['config'] = [config_map] * len(df)

                all_data.append(df)
                processed_seeds += 1
                total_rows += len(df)

            except Exception as e:
                error_msg = f"Error processing {parquet_path}: {e}"
                print(f"\n  ERROR: {error_msg}")
                errors.append(error_msg)
                continue

        processed_experiments += 1

    print(f"\n\nProcessed: {processed_experiments}/{len(experiments)} experiments, " +
          f"{processed_seeds}/{total_seeds} seeds, {total_rows:,} rows")

    if errors:
        print(f"\nEncountered {len(errors)} errors during processing:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    if not all_data:
        print("\nERROR: No data to consolidate")
        return False

    # Concatenate all data
    print("\n" + "-" * 80)
    print("PHASE 3: Concatenating data")
    print("-" * 80)

    combined = pd.concat(all_data, ignore_index=True)
    print(f"Combined shape: {combined.shape}")
    memory_gb = combined.memory_usage(deep=True).sum() / (1024**3)
    print(f"Memory usage: {memory_gb:.2f} GB")

    # Sort by exp_name, exp_hash, seed, t_submit
    print("\n" + "-" * 80)
    print("PHASE 4: Sorting data")
    print("-" * 80)
    print("Sorting by: exp_name, exp_hash, seed, t_submit")

    combined = combined.sort_values(['exp_name', 'exp_hash', 'seed', 't_submit'])
    combined = combined.reset_index(drop=True)

    # Define schema with MAP type
    print("\n" + "-" * 80)
    print("PHASE 5: Converting to PyArrow table")
    print("-" * 80)

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

    try:
        table = pa.Table.from_pandas(combined, schema=schema)
        print(f"Table shape: {table.shape}")
        print(f"Table size in memory: {table.nbytes / (1024**3):.2f} GB")
    except Exception as e:
        print(f"\nERROR: Failed to create PyArrow table: {e}")
        return False

    # Write to Parquet with zstd compression
    print("\n" + "-" * 80)
    print("PHASE 6: Writing consolidated Parquet file")
    print("-" * 80)
    print(f"Output: {output_path}")
    print(f"Compression: {compression} (level {compression_level})")

    try:
        pq.write_table(
            table,
            output_path,
            compression=compression,
            compression_level=compression_level if compression == 'zstd' else None,
            use_dictionary=['exp_name', 'exp_hash', 'status'],
            write_statistics=True,
            row_group_size=None,  # Auto-size (will be ~one per seed)
            version='2.6'
        )
    except Exception as e:
        print(f"\nERROR: Failed to write Parquet file: {e}")
        return False

    # Print summary
    file_size = Path(output_path).stat().st_size
    compression_ratio = memory_gb * (1024**3) / file_size

    print("\n" + "=" * 80)
    print("CONSOLIDATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")
    print(f"File size: {file_size / (1024**2):.1f} MB ({file_size / (1024**3):.2f} GB)")
    print(f"Rows: {len(combined):,}")
    print(f"Experiments: {processed_experiments}")
    print(f"Seeds: {processed_seeds}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"\n✓ Original files PRESERVED (not deleted)")
    print(f"  Location: {base_dir}/exp*/[0-9]*/results.parquet")
    print(f"  Count: {total_seeds} files")
    print(f"\nNext steps:")
    print(f"  1. Verify consolidation: python scripts/verify_consolidation.py")
    print(f"  2. Test analysis with consolidated format")
    print(f"  3. Compare results with per-seed format")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Consolidate all experiments into single Parquet file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be done
  python scripts/consolidate_all_experiments.py --dry-run

  # Run consolidation with default settings (zstd level 3)
  python scripts/consolidate_all_experiments.py

  # Use maximum compression (slower)
  python scripts/consolidate_all_experiments.py --compression-level 9

  # Use snappy instead of zstd (faster, larger file)
  python scripts/consolidate_all_experiments.py --compression snappy

IMPORTANT: Original files are NEVER deleted. They are preserved for safety.
        """
    )
    parser.add_argument('--base-dir', default='experiments',
                       help='Base directory containing experiments (default: experiments)')
    parser.add_argument('--output', default='experiments/consolidated.parquet',
                       help='Output file path (default: experiments/consolidated.parquet)')
    parser.add_argument('--compression', default='zstd',
                       choices=['zstd', 'snappy', 'gzip', 'brotli'],
                       help='Compression codec (default: zstd)')
    parser.add_argument('--compression-level', type=int, default=3,
                       help='Compression level for zstd (1-22, default: 3)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without writing')

    args = parser.parse_args()

    success = consolidate_all_experiments(
        base_dir=args.base_dir,
        output_path=args.output,
        compression=args.compression,
        compression_level=args.compression_level,
        dry_run=args.dry_run
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
