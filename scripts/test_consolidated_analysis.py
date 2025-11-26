#!/usr/bin/env python3
"""
Test script to verify consolidated file produces identical analysis results.

Compares results from:
1. Original method (individual files)
2. Consolidated method (single file with predicate pushdown)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from icecap import saturation_analysis


def test_single_experiment(exp_info: dict, consolidated_path: str) -> bool:
    """Test that both methods produce identical results for one experiment."""

    exp_label = f"{exp_info['label']}-{exp_info['hash']}"
    print(f"\n{'='*70}")
    print(f"Testing: {exp_label}")
    print('='*70)

    # Load using original method
    print("Loading with original method (individual files)...")
    df_original = saturation_analysis.load_and_aggregate_results(exp_info)

    # Load using consolidated method
    print("Loading with consolidated method (predicate pushdown)...")
    df_consolidated = saturation_analysis.load_and_aggregate_results_consolidated(
        exp_info, consolidated_path
    )

    # Normalize schemas before comparison
    if df_original is not None:
        # Convert seed column to string for both (original has string, consolidated has int)
        df_original['seed'] = df_original['seed'].astype(str)

        # Normalize time columns to int64 for both (for fair comparison)
        time_cols = ['t_commit', 'commit_latency', 'total_latency']
        for col in time_cols:
            if col in df_original.columns and df_original[col].dtype == 'float64':
                df_original[col] = df_original[col].round().astype('int64')

    if df_consolidated is not None:
        # Convert seed to string to match original
        df_consolidated['seed'] = df_consolidated['seed'].astype(str)

    # Both should be None or both should have data
    if df_original is None and df_consolidated is None:
        print("✓ Both methods returned None (no data)")
        return True

    if df_original is None or df_consolidated is None:
        print(f"❌ MISMATCH: One method returned None")
        print(f"   Original: {'None' if df_original is None else f'{len(df_original)} rows'}")
        print(f"   Consolidated: {'None' if df_consolidated is None else f'{len(df_consolidated)} rows'}")
        return False

    # Compare shapes
    if df_original.shape != df_consolidated.shape:
        print(f"❌ MISMATCH: Different shapes")
        print(f"   Original: {df_original.shape}")
        print(f"   Consolidated: {df_consolidated.shape}")
        return False

    print(f"  Shape: {df_original.shape} ✓")

    # Compare column names (consolidated has fewer columns)
    orig_cols = set(df_original.columns)
    cons_cols = set(df_consolidated.columns)

    if orig_cols != cons_cols:
        print(f"❌ MISMATCH: Different columns")
        print(f"   Original only: {orig_cols - cons_cols}")
        print(f"   Consolidated only: {cons_cols - orig_cols}")
        return False

    print(f"  Columns: {len(orig_cols)} columns ✓")

    # Sort both by same columns for comparison
    sort_cols = ['seed', 't_submit', 'txn_id']
    df_orig_sorted = df_original.sort_values(sort_cols).reset_index(drop=True)
    df_cons_sorted = df_consolidated.sort_values(sort_cols).reset_index(drop=True)

    # Compare values column by column
    mismatches = []
    for col in df_orig_sorted.columns:
        if not df_orig_sorted[col].equals(df_cons_sorted[col]):
            # For numeric columns, check if they're close
            if pd.api.types.is_numeric_dtype(df_orig_sorted[col]):
                if not (df_orig_sorted[col] == df_cons_sorted[col]).all():
                    diffs = (df_orig_sorted[col] != df_cons_sorted[col]).sum()
                    mismatches.append(f"{col}: {diffs} differences")
            else:
                mismatches.append(f"{col}: values don't match")

    if mismatches:
        print(f"❌ MISMATCH in columns:")
        for m in mismatches:
            print(f"   - {m}")
        return False

    print(f"  Values: All columns match ✓")

    # Compare aggregated statistics
    orig_committed = df_orig_sorted[df_orig_sorted['status'] == 'committed']
    cons_committed = df_cons_sorted[df_cons_sorted['status'] == 'committed']

    orig_commit_rate = len(orig_committed) / len(df_orig_sorted)
    cons_commit_rate = len(cons_committed) / len(df_cons_sorted)

    if abs(orig_commit_rate - cons_commit_rate) > 0.0001:
        print(f"❌ MISMATCH: Commit rates differ")
        print(f"   Original: {orig_commit_rate:.4f}")
        print(f"   Consolidated: {cons_commit_rate:.4f}")
        return False

    print(f"  Commit rate: {orig_commit_rate:.2%} ✓")

    if len(orig_committed) > 0:
        orig_mean_latency = orig_committed['commit_latency'].mean()
        cons_mean_latency = cons_committed['commit_latency'].mean()

        if abs(orig_mean_latency - cons_mean_latency) > 1.0:  # Within 1ms
            print(f"❌ MISMATCH: Mean latencies differ")
            print(f"   Original: {orig_mean_latency:.2f}ms")
            print(f"   Consolidated: {cons_mean_latency:.2f}ms")
            return False

        print(f"  Mean latency: {orig_mean_latency:.1f}ms ✓")

    print(f"\n✓ PASSED: {exp_label}")
    return True


def main():
    """Run comparison test on a sample of experiments."""
    import random
    import argparse

    parser = argparse.ArgumentParser(description='Test consolidated analysis')
    parser.add_argument('--sample', type=int, default=10,
                       help='Number of experiments to test')
    parser.add_argument('--pattern', default='exp2_1*',
                       help='Experiment pattern to test')
    parser.add_argument('--consolidated', default='experiments/consolidated.parquet',
                       help='Path to consolidated file')

    args = parser.parse_args()

    print("="*70)
    print("CONSOLIDATED ANALYSIS TEST")
    print("="*70)
    print(f"Pattern: {args.pattern}")
    print(f"Sample size: {args.sample}")
    print(f"Consolidated file: {args.consolidated}")

    # Load configuration
    saturation_analysis.CONFIG = saturation_analysis.get_default_config()

    # Discover experiments
    experiments = saturation_analysis.scan_experiment_directories('experiments', args.pattern)

    if not experiments:
        print(f"\nNo experiments found matching pattern: {args.pattern}")
        return 1

    print(f"\nFound {len(experiments)} experiments")

    # Sample experiments to test
    exp_list = list(experiments.items())
    if len(exp_list) > args.sample:
        exp_list = random.sample(exp_list, args.sample)
        print(f"Testing random sample of {args.sample} experiments")
    else:
        print(f"Testing all {len(exp_list)} experiments")

    # Run tests
    passed = 0
    failed = 0

    for exp_dir, exp_info in exp_list:
        try:
            if test_single_experiment(exp_info, args.consolidated):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ ERROR testing {exp_info['label']}-{exp_info['hash']}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    if failed == 0:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())
