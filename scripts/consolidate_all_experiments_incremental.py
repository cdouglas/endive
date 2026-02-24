#!/usr/bin/env python3
"""
Memory-efficient consolidation using incremental PyArrow writing.

This version processes experiments in batches and appends to the Parquet file
incrementally, avoiding memory exhaustion.

Includes optional verification to validate consolidated data matches originals.
"""

import argparse
import random
import shutil
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

    # Fill NaN in string columns (e.g., abort_reason is null for committed txns)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('')

    # Normalize count columns: int64/int32 → int8
    count_cols = ['n_retries']
    for col in count_cols:
        if col in df.columns and df[col].dtype in ('int64', 'int32'):
            if df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')

    return df


def discover_schema(experiments):
    """Discover schema from first available results.parquet."""
    for _, _, _, seed_dirs in experiments:
        for seed_dir in seed_dirs:
            parquet_path = seed_dir / 'results.parquet'
            if parquet_path.exists():
                df = load_and_normalize_schema(str(parquet_path))
                table = pa.Table.from_pandas(df, preserve_index=False)
                fields = list(table.schema)
                fields.extend([
                    pa.field('exp_name', pa.string()),
                    pa.field('exp_hash', pa.string()),
                    pa.field('seed', pa.int64()),
                    pa.field('config', pa.map_(pa.string(), pa.string())),
                ])
                return pa.schema(fields)
    raise ValueError("No results.parquet files found to discover schema")


def consolidate_incremental(
    base_dir: str = 'experiments',
    output_path: str = 'experiments/consolidated.parquet',
    batch_size: int = 50,
    compression: str = 'zstd',
    compression_level: int = 3,
    destructive: bool = False
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
    if destructive:
        print(f"Mode: DESTRUCTIVE (experiment directories deleted after writing)")
        print(f"\n  ⚠ WARNING: Original experiment files will be deleted as they are")
        print(f"  consolidated. If this process is interrupted, any already-deleted")
        print(f"  experiments will only exist in the partially-written consolidated file.")

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

    # Discover schema from first available results.parquet
    schema = discover_schema(experiments)

    # Process experiments in batches and write incrementally
    print("\n" + "-" * 80)
    print("PHASE 2: Processing and writing incrementally")
    print("-" * 80)

    writer = None
    total_rows = 0
    deleted_dirs = 0
    deleted_bytes = 0

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

            # Track whether all seeds for this experiment succeeded
            exp_seeds_ok = True

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
                    exp_seeds_ok = False
                    continue

            # Delete experiment directory after all seeds written
            if destructive and exp_seeds_ok:
                dir_size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
                shutil.rmtree(exp_dir)
                deleted_dirs += 1
                deleted_bytes += dir_size

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
        if destructive:
            print(f"\n✗ Deleted {deleted_dirs} experiment directories"
                  f" ({deleted_bytes / (1024**2):.1f} MB freed)")
        else:
            print(f"\n✓ Original files PRESERVED (not deleted)")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataframes_match(original_df: pd.DataFrame, consolidated_df: pd.DataFrame,
                            exp_name: str, exp_hash: str, seed: int) -> bool:
    """Compare two dataframes and return True if they match."""
    # Compare on shared columns only (consolidated may have extra metadata columns)
    shared_cols = sorted(set(original_df.columns) & set(consolidated_df.columns))
    original_df = original_df[shared_cols].reset_index(drop=True)
    consolidated_df = consolidated_df[shared_cols].reset_index(drop=True)

    # Compare shapes
    if original_df.shape != consolidated_df.shape:
        print(f"  ❌ {exp_name}-{exp_hash}/{seed}: Shape mismatch: "
              f"original {original_df.shape} vs consolidated {consolidated_df.shape}")
        return False

    # Compare values column by column
    for col in original_df.columns:

        if not original_df[col].equals(consolidated_df[col]):
            if pd.api.types.is_numeric_dtype(original_df[col]):
                if not (original_df[col] == consolidated_df[col]).all():
                    diffs = (original_df[col] != consolidated_df[col]).sum()
                    print(f"  ❌ {exp_name}-{exp_hash}/{seed}: Column '{col}' has {diffs} differences")
                    return False
            else:
                print(f"  ❌ {exp_name}-{exp_hash}/{seed}: Column '{col}' values don't match")
                return False

    return True


def verify_consolidation(
    consolidated_path: str,
    base_dir: str = 'experiments',
    sample_size: int = 20
) -> Tuple[int, int]:
    """
    Verify consolidated data matches original files.

    Uses predicate pushdown to efficiently seek within consolidated file.

    Returns:
        Tuple of (passed, failed) counts
    """
    print("\n" + "-" * 80)
    print("PHASE 3: Verification")
    print("-" * 80)

    # Find all results.parquet files
    all_files = list(Path(base_dir).glob('*-*/[0-9]*/results.parquet'))
    if not all_files:
        print("No result files found to verify")
        return 0, 0

    # Sample files to verify
    files_to_verify = random.sample(all_files, min(sample_size, len(all_files)))
    print(f"\nVerifying random sample of {len(files_to_verify)} / {len(all_files)} files...")

    passed = 0
    failed = 0

    for file_path in tqdm(files_to_verify, desc="Verifying", unit="file"):
        # Extract metadata from path
        seed_dir = file_path.parent
        exp_dir = seed_dir.parent
        seed = int(seed_dir.name)
        dir_name = exp_dir.name

        if '-' not in dir_name:
            continue

        exp_name, exp_hash = dir_name.rsplit('-', 1)

        try:
            # Load original with same normalization as consolidation
            original_df = load_and_normalize_schema(str(file_path))
            original_df = original_df.sort_values('t_submit').reset_index(drop=True)

            # Load matching slice from consolidated using predicate pushdown
            consolidated_df = pd.read_parquet(
                consolidated_path,
                filters=[
                    ('exp_name', '==', exp_name),
                    ('exp_hash', '==', exp_hash),
                    ('seed', '==', seed)
                ]
            )

            if len(consolidated_df) == 0:
                tqdm.write(f"  ❌ {exp_name}-{exp_hash}/{seed}: Not found in consolidated")
                failed += 1
                continue

            # Drop metadata columns and reset index
            consolidated_df = consolidated_df.drop(columns=['exp_name', 'exp_hash', 'seed', 'config'])
            consolidated_df = consolidated_df.reset_index(drop=True)

            # Compare
            if verify_dataframes_match(original_df, consolidated_df, exp_name, exp_hash, seed):
                passed += 1
            else:
                failed += 1

        except Exception as e:
            tqdm.write(f"  ❌ {exp_name}-{exp_hash}/{seed}: Error - {e}")
            failed += 1

    # Summary
    print(f"\nVerification: {passed} passed, {failed} failed")
    if failed == 0:
        print("✓ All sampled files match consolidated data")
    else:
        print("❌ Some files failed verification")

    return passed, failed


def main():
    parser = argparse.ArgumentParser(description='Memory-efficient consolidation')
    parser.add_argument('--base-dir', default='experiments')
    parser.add_argument('--output', default='experiments/consolidated.parquet')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--compression', default='zstd', choices=['zstd', 'snappy'])
    parser.add_argument('--compression-level', type=int, default=3)
    parser.add_argument('--verify', action='store_true',
                        help='Verify consolidated data matches originals after writing')
    parser.add_argument('--verify-sample', type=int, default=20,
                        help='Number of random files to verify (default: 20)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only run verification (skip consolidation)')
    parser.add_argument('--destructive', action='store_true',
                        help='Delete experiment directories after writing to consolidated file. '
                             'Frees disk space incrementally but risks data loss if interrupted.')

    args = parser.parse_args()

    # Verify-only mode
    if args.verify_only:
        passed, failed = verify_consolidation(
            args.output,
            args.base_dir,
            args.verify_sample
        )
        sys.exit(0 if failed == 0 else 1)

    # Normal consolidation
    success = consolidate_incremental(
        base_dir=args.base_dir,
        output_path=args.output,
        batch_size=args.batch_size,
        compression=args.compression,
        compression_level=args.compression_level,
        destructive=args.destructive
    )

    if not success:
        sys.exit(1)

    # Optional verification after consolidation
    if args.verify:
        passed, failed = verify_consolidation(
            args.output,
            args.base_dir,
            args.verify_sample
        )
        if failed > 0:
            sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
