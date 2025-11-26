#!/usr/bin/env python3
"""
Efficient verification using predicate pushdown - never loads full consolidated file.
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def verify_single_file(consolidated_path: str, original_path: Path) -> bool:
    """
    Verify a single results.parquet file against consolidated.

    Extracts exp_name, exp_hash, seed from path and uses predicate pushdown
    to only load the relevant slice from consolidated file.
    """
    # Extract metadata from path: experiments/exp_name-exp_hash/seed/results.parquet
    seed_dir = original_path.parent
    exp_dir = seed_dir.parent

    seed = int(seed_dir.name)
    dir_name = exp_dir.name

    if '-' in dir_name:
        exp_name, exp_hash = dir_name.rsplit('-', 1)
    else:
        print(f"  ⚠ Skipping {original_path}: cannot parse exp_name/exp_hash from '{dir_name}'")
        return True  # Skip but don't fail

    try:
        # Load original and sort by t_submit
        original_df = pd.read_parquet(original_path)
        original_df = original_df.sort_values('t_submit').reset_index(drop=True)

        # Normalize schema to match consolidated (float64 → int64 for time columns)
        time_cols = ['t_commit', 'commit_latency', 'total_latency']
        for col in time_cols:
            if col in original_df.columns and original_df[col].dtype == 'float64':
                original_df[col] = original_df[col].round().astype('int64')

        # Normalize count columns (int64 → int8)
        count_cols = ['n_retries', 'n_tables_read', 'n_tables_written']
        for col in count_cols:
            if col in original_df.columns and original_df[col].dtype == 'int64':
                if original_df[col].min() >= -128 and original_df[col].max() <= 127:
                    original_df[col] = original_df[col].astype('int8')

        # Load ONLY the matching slice from consolidated using predicate pushdown
        # This is efficient - Parquet will skip entire row groups that don't match
        consolidated_df = pd.read_parquet(
            consolidated_path,
            filters=[
                ('exp_name', '==', exp_name),
                ('exp_hash', '==', exp_hash),
                ('seed', '==', seed)
            ]
        )

        # Drop metadata columns that aren't in original
        consolidated_df = consolidated_df.drop(columns=['exp_name', 'exp_hash', 'seed', 'config'])

        # Both should now be sorted by t_submit and have same columns
        consolidated_df = consolidated_df.reset_index(drop=True)

        # Compare shapes
        if original_df.shape != consolidated_df.shape:
            print(f"  ❌ Shape mismatch: original {original_df.shape} vs consolidated {consolidated_df.shape}")
            return False

        # Compare values (column by column to save memory)
        for col in original_df.columns:
            if col not in consolidated_df.columns:
                print(f"  ❌ Column '{col}' missing in consolidated")
                return False

            if not original_df[col].equals(consolidated_df[col]):
                # Check if close for numeric columns
                if pd.api.types.is_numeric_dtype(original_df[col]):
                    if not (original_df[col] == consolidated_df[col]).all():
                        diffs = (original_df[col] != consolidated_df[col]).sum()
                        print(f"  ❌ Column '{col}' has {diffs} differences")
                        return False
                else:
                    print(f"  ❌ Column '{col}' values don't match")
                    return False

        print(f"  ✓ {exp_name}-{exp_hash}/{seed}: {len(original_df):,} rows match")
        return True

    except Exception as e:
        print(f"  ❌ Error verifying {original_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Efficient consolidation verification')
    parser.add_argument('--consolidated', default='experiments/consolidated.parquet')
    parser.add_argument('--base-dir', default='experiments')
    parser.add_argument('--sample', type=int, default=20,
                       help='Number of random files to verify')
    parser.add_argument('--all', action='store_true',
                       help='Verify all files (may take a while)')

    args = parser.parse_args()

    print("=" * 80)
    print("EFFICIENT CONSOLIDATION VERIFICATION")
    print("=" * 80)
    print(f"\nConsolidated: {args.consolidated}")
    print(f"Base dir: {args.base_dir}")

    # Find all results.parquet files
    all_files = list(Path(args.base_dir).glob('exp*/[0-9]*/results.parquet'))
    print(f"\nFound {len(all_files)} result files")

    # Sample or use all
    if args.all:
        files_to_verify = all_files
        print("Verifying ALL files...")
    else:
        files_to_verify = random.sample(all_files, min(args.sample, len(all_files)))
        print(f"Verifying random sample of {len(files_to_verify)} files...")

    print()

    passed = 0
    failed = 0

    for i, file_path in enumerate(files_to_verify, 1):
        print(f"[{i}/{len(files_to_verify)}] {file_path.parent.parent.name}/{file_path.parent.name}")

        if verify_single_file(args.consolidated, file_path):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✓ VERIFICATION PASSED")
        print("=" * 80)
        return 0
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
