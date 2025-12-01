#!/usr/bin/env python3
"""
Adaptive consolidation verification that switches strategies based on sample size.

Two strategies:
1. Small sample (< 1/4 corpus): Load individual results.parquet files and seek within consolidated
2. Large sample (>= 1/4 corpus): Full scan consolidated.parquet and look up results.parquet files
"""

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def verify_small_sample(consolidated_path: str, files_to_verify: list[Path]) -> tuple[int, int]:
    """
    Strategy 1: For each results.parquet, seek within consolidated using predicate pushdown.
    Efficient when verifying a small sample.
    """
    passed = 0
    failed = 0

    for file_path in tqdm(files_to_verify, desc="Verifying", unit="file"):
        # Extract metadata from path
        seed_dir = file_path.parent
        exp_dir = seed_dir.parent
        seed = int(seed_dir.name)
        dir_name = exp_dir.name

        if '-' in dir_name:
            exp_name, exp_hash = dir_name.rsplit('-', 1)
        else:
            tqdm.write(f"  ⚠ Skipping {file_path}: cannot parse exp_name/exp_hash from '{dir_name}'")
            continue

        tqdm.write(f"  ✓ {exp_name}-{exp_hash}/{seed}")

        try:
            # Load original and sort by t_submit
            original_df = pd.read_parquet(file_path)
            original_df = original_df.sort_values('t_submit').reset_index(drop=True)

            # Normalize schema to match consolidated
            time_cols = ['t_commit', 'commit_latency', 'total_latency']
            for col in time_cols:
                if col in original_df.columns and original_df[col].dtype == 'float64':
                    original_df[col] = original_df[col].round().astype('int64')

            count_cols = ['n_retries', 'n_tables_read', 'n_tables_written']
            for col in count_cols:
                if col in original_df.columns and original_df[col].dtype == 'int64':
                    if original_df[col].min() >= -128 and original_df[col].max() <= 127:
                        original_df[col] = original_df[col].astype('int8')

            # Load matching slice from consolidated using predicate pushdown
            consolidated_df = pd.read_parquet(
                consolidated_path,
                filters=[
                    ('exp_name', '==', exp_name),
                    ('exp_hash', '==', exp_hash),
                    ('seed', '==', seed)
                ]
            )

            # Drop metadata columns (consolidated is already sorted by t_submit - don't re-sort!)
            consolidated_df = consolidated_df.drop(columns=['exp_name', 'exp_hash', 'seed', 'config'])
            consolidated_df = consolidated_df.reset_index(drop=True)

            # Compare
            if not verify_dataframes_match(original_df, consolidated_df, exp_name, exp_hash, seed):
                failed += 1
            else:
                passed += 1

        except Exception as e:
            tqdm.write(f"  ❌ Error: {e}")
            failed += 1

    return passed, failed


def verify_large_sample(consolidated_path: str, files_to_verify: list[Path], base_dir: str) -> tuple[int, int]:
    """
    Strategy 2: Single-pass stream through consolidated.parquet.
    Verifies experiments as soon as they're complete (seeds are contiguous).
    Memory-efficient: only holds current row group + 1-2 experiments in transition.
    """
    print("\nUsing full-scan strategy (sample >= 1/4 of corpus)")
    print("Streaming through consolidated file...")

    # Build set of files to verify for quick lookup
    files_set = set(files_to_verify)
    print(f"Sample contains {len(files_set)} files to verify")

    # Open parquet file for streaming
    parquet_file = pq.ParquetFile(consolidated_path)
    total_row_groups = parquet_file.num_row_groups

    print(f"Consolidated file has {total_row_groups} row groups")
    print()

    # Track validated files
    validated_files = set()
    passed = 0
    failed = 0

    # Current experiment being accumulated (seeds are contiguous in file)
    current_key = None
    current_chunks = []

    def verify_current():
        """Verify the current accumulated experiment and clear memory."""
        nonlocal current_key, current_chunks, passed, failed

        if current_key is None:
            return

        exp_name, exp_hash, seed = current_key
        results_path = Path(base_dir) / f"{exp_name}-{exp_hash}" / str(seed) / "results.parquet"

        # Only verify if in sample
        if results_path not in files_set:
            current_key = None
            current_chunks = []
            return

        try:
            # Concatenate accumulated chunks (already sorted by t_submit in consolidated file)
            consolidated_df = pd.concat(current_chunks, ignore_index=True)
            consolidated_df = consolidated_df.drop(columns=['exp_name', 'exp_hash', 'seed', 'config'])
            consolidated_df = consolidated_df.reset_index(drop=True)

            # Load and normalize original (small file, < 9 MiB)
            if not results_path.exists():
                tqdm.write(f"  ❌ {exp_name}-{exp_hash}/{seed}: File not found")
                failed += 1
                current_key = None
                current_chunks = []
                return

            original_df = pd.read_parquet(results_path)
            original_df = original_df.sort_values('t_submit').reset_index(drop=True)

            # Normalize schema
            time_cols = ['t_commit', 'commit_latency', 'total_latency']
            for col in time_cols:
                if col in original_df.columns and original_df[col].dtype == 'float64':
                    original_df[col] = original_df[col].round().astype('int64')

            count_cols = ['n_retries', 'n_tables_read', 'n_tables_written']
            for col in count_cols:
                if col in original_df.columns and original_df[col].dtype == 'int64':
                    if original_df[col].min() >= -128 and original_df[col].max() <= 127:
                        original_df[col] = original_df[col].astype('int8')

            # Compare
            if verify_dataframes_match(original_df, consolidated_df, exp_name, exp_hash, seed):
                validated_files.add(results_path)
                passed += 1
            else:
                failed += 1

        except Exception as e:
            tqdm.write(f"  ❌ {exp_name}-{exp_hash}/{seed}: {e}")
            failed += 1

        # Clear memory
        current_key = None
        current_chunks = []

    # Single-pass stream
    for rg_idx in tqdm(range(total_row_groups), desc="Streaming & verifying", unit="rg"):
        table = parquet_file.read_row_group(rg_idx)
        batch_df = table.to_pandas()

        if len(batch_df) == 0:
            continue

        # Process experiments in this row group
        for (exp_name, exp_hash, seed), group in batch_df.groupby(['exp_name', 'exp_hash', 'seed']):
            key = (exp_name, exp_hash, seed)

            # If we've moved to a different experiment, verify the previous one
            if current_key is not None and key != current_key:
                verify_current()

            # Accumulate chunk for current experiment
            current_key = key
            current_chunks.append(group)

    # Verify final experiment
    verify_current()

    # Check for files in sample that weren't found in consolidated
    missing = files_set - validated_files
    if missing:
        print(f"\n⚠ WARNING: {len(missing)} files in sample not found in consolidated:")
        for path in sorted(missing)[:5]:
            print(f"  - {path.parent.parent.name}/{path.parent.name}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    return passed, failed


def verify_dataframes_match(original_df: pd.DataFrame, consolidated_df: pd.DataFrame,
                            exp_name: str, exp_hash: str, seed: int) -> bool:
    """Compare two dataframes and return True if they match."""
    # Compare shapes
    if original_df.shape != consolidated_df.shape:
        print(f"  ❌ Shape mismatch: original {original_df.shape} vs consolidated {consolidated_df.shape}")
        return False

    # Compare values column by column
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

    # print(f"  ✓ {exp_name}-{exp_hash}/{seed}: {len(original_df):,} rows match")
    return True


def main():
    parser = argparse.ArgumentParser(description='Adaptive consolidation verification')
    parser.add_argument('--consolidated', default='experiments/consolidated.parquet')
    parser.add_argument('--base-dir', default='experiments')
    parser.add_argument('--sample', type=int, default=20,
                       help='Number of random files to verify')
    parser.add_argument('--all', action='store_true',
                       help='Verify all files (may take a while)')

    args = parser.parse_args()

    print("=" * 80)
    print("ADAPTIVE CONSOLIDATION VERIFICATION")
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

    # Determine strategy based on sample size
    corpus_size = len(all_files)
    sample_size = len(files_to_verify)
    threshold = corpus_size // 4

    print(f"\nSample size: {sample_size}")
    print(f"Corpus size: {corpus_size}")
    print(f"Threshold (1/4): {threshold}")

    if sample_size >= threshold:
        print("\n→ Using FULL-SCAN strategy (sample >= 1/4 of corpus)")
        print("  This scans the consolidated file once and looks up individual files")
        passed, failed = verify_large_sample(args.consolidated, files_to_verify, args.base_dir)
    else:
        print("\n→ Using SEEK strategy (sample < 1/4 of corpus)")
        print("  This opens individual files and seeks within consolidated")
        print()
        passed, failed = verify_small_sample(args.consolidated, files_to_verify)

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
