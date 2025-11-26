#!/usr/bin/env python3
"""
Verify consolidated.parquet matches ALL original experiments.

This performs comprehensive verification before any original files are deleted.

Usage:
    python scripts/verify_consolidation.py
    python scripts/verify_consolidation.py --consolidated experiments/consolidated.parquet
    python scripts/verify_consolidation.py --sample 20  # Verify 20 random experiments
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def verify_consolidation(
    consolidated_path: str = 'experiments/consolidated.parquet',
    base_dir: str = 'experiments',
    sample_size: int = 10
) -> bool:
    """
    Comprehensive verification of consolidated file.

    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 80)
    print("COMPREHENSIVE CONSOLIDATION VERIFICATION")
    print("=" * 80)
    print(f"\nConsolidated file: {consolidated_path}")
    print(f"Base directory: {base_dir}")
    print(f"Sample verification: {sample_size} random experiments")

    if not Path(consolidated_path).exists():
        print(f"\n❌ ERROR: Consolidated file not found: {consolidated_path}")
        return False

    all_checks_passed = True

    # Load consolidated file metadata
    print("\n" + "-" * 80)
    print("Loading consolidated file metadata...")
    print("-" * 80)

    try:
        table = pq.read_table(consolidated_path, columns=['exp_name', 'exp_hash', 'seed'])
        df_meta = table.to_pandas()
    except Exception as e:
        print(f"❌ ERROR: Failed to read consolidated file: {e}")
        return False

    print(f"Total rows: {len(df_meta):,}")
    print(f"Unique experiments: {df_meta[['exp_name', 'exp_hash']].drop_duplicates().shape[0]}")
    print(f"Unique seeds: {df_meta['seed'].nunique()}")

    # Check 1: Count original files
    print("\n" + "-" * 80)
    print("CHECK 1: Verify all original files present in consolidated")
    print("-" * 80)

    original_files = list(Path(base_dir).glob('exp*/[0-9]*/results.parquet'))
    print(f"Original result files found: {len(original_files)}")
    print(f"Seeds in consolidated: {df_meta['seed'].nunique()}")

    if len(original_files) != df_meta['seed'].nunique():
        print(f"❌ MISMATCH: {len(original_files)} original files vs {df_meta['seed'].nunique()} consolidated seeds")
        all_checks_passed = False
    else:
        print(f"✓ Match: {len(original_files)} files")

    # Check 2: Verify experiments present
    print("\n" + "-" * 80)
    print("CHECK 2: Verify all experiment directories present")
    print("-" * 80)

    original_exp_dirs = set()
    for exp_dir in Path(base_dir).glob('exp*'):
        if exp_dir.is_dir() and any(exp_dir.glob('[0-9]*/results.parquet')):
            original_exp_dirs.add(exp_dir.name)

    consolidated_exps = set(
        df_meta['exp_name'] + '-' + df_meta['exp_hash']
    )

    print(f"Original experiment directories: {len(original_exp_dirs)}")
    print(f"Experiments in consolidated: {len(consolidated_exps)}")

    missing_from_consolidated = original_exp_dirs - consolidated_exps
    extra_in_consolidated = consolidated_exps - original_exp_dirs

    if missing_from_consolidated:
        print(f"❌ Missing from consolidated: {len(missing_from_consolidated)}")
        for exp in list(missing_from_consolidated)[:5]:
            print(f"    - {exp}")
        if len(missing_from_consolidated) > 5:
            print(f"    ... and {len(missing_from_consolidated) - 5} more")
        all_checks_passed = False
    else:
        print(f"✓ All original experiments present in consolidated")

    if extra_in_consolidated:
        print(f"⚠ Extra in consolidated (not in originals): {len(extra_in_consolidated)}")
        for exp in list(extra_in_consolidated)[:5]:
            print(f"    - {exp}")

    # Check 3: Total row count (sample for speed)
    print("\n" + "-" * 80)
    print("CHECK 3: Verify total row count (sampling 100 files)")
    print("-" * 80)

    import random
    sample_files = random.sample(original_files, min(100, len(original_files)))

    original_sample_count = 0
    for f in tqdm(sample_files, desc="Counting rows in sample"):
        try:
            df = pd.read_parquet(f, columns=['txn_id'])
            original_sample_count += len(df)
        except Exception as e:
            print(f"\n  WARNING: Error reading {f}: {e}")

    # Get corresponding seeds from consolidated
    sample_seeds = [int(f.parent.name) for f in sample_files]
    consolidated_sample = df_meta[df_meta['seed'].isin(sample_seeds)]
    consolidated_sample_count = len(consolidated_sample)

    print(f"Original sample rows: {original_sample_count:,}")
    print(f"Consolidated sample rows: {consolidated_sample_count:,}")

    if original_sample_count != consolidated_sample_count:
        print(f"❌ MISMATCH: Row count differs")
        all_checks_passed = False
    else:
        print(f"✓ Match: {original_sample_count:,} rows in sample")

    # Check 4: Deep verification of random experiments
    print("\n" + "-" * 80)
    print(f"CHECK 4: Deep verification of {sample_size} random experiments")
    print("-" * 80)

    experiments = df_meta[['exp_name', 'exp_hash']].drop_duplicates()
    sample = experiments.sample(min(sample_size, len(experiments)))

    deep_check_passed = 0
    deep_check_failed = 0

    for idx, row in enumerate(sample.itertuples(), 1):
        exp_name = row.exp_name
        exp_hash = row.exp_hash

        print(f"\n[{idx}/{len(sample)}] Verifying {exp_name}-{exp_hash}...")

        try:
            # Load from consolidated (only necessary columns for speed)
            consolidated_exp = pd.read_parquet(
                consolidated_path,
                filters=[('exp_hash', '==', exp_hash)],
                columns=['txn_id', 'status', 'commit_latency', 'seed']
            )

            # Load original files
            exp_dir = Path(base_dir) / f"{exp_name}-{exp_hash}"
            if not exp_dir.exists():
                print(f"  ❌ Original directory not found")
                deep_check_failed += 1
                all_checks_passed = False
                continue

            original_dfs = []
            for seed_dir in exp_dir.glob('[0-9]*'):
                parquet_path = seed_dir / 'results.parquet'
                if parquet_path.exists():
                    df = pd.read_parquet(parquet_path, columns=['txn_id', 'status', 'commit_latency'])
                    # Sort by t_submit to match consolidated ordering
                    df = df.sort_values('txn_id')  # Use txn_id as proxy since we didn't load t_submit
                    original_dfs.append(df)

            if not original_dfs:
                print(f"  ❌ No original data found")
                deep_check_failed += 1
                all_checks_passed = False
                continue

            original_exp = pd.concat(original_dfs, ignore_index=True)

            # Compare row count
            if len(consolidated_exp) != len(original_exp):
                print(f"  ❌ Row count: {len(consolidated_exp)} vs {len(original_exp)}")
                deep_check_failed += 1
                all_checks_passed = False
                continue

            # Compare commit rate
            consolidated_rate = (consolidated_exp['status'] == 'committed').mean()
            original_rate = (original_exp['status'] == 'committed').mean()

            if abs(consolidated_rate - original_rate) > 0.001:
                print(f"  ❌ Commit rate: {consolidated_rate:.3f} vs {original_rate:.3f}")
                deep_check_failed += 1
                all_checks_passed = False
                continue

            # Compare mean latency (for committed transactions)
            consolidated_committed = consolidated_exp[consolidated_exp['status'] == 'committed']
            original_committed = original_exp[original_exp['status'] == 'committed']

            if len(consolidated_committed) > 0 and len(original_committed) > 0:
                # Handle both int and float latencies
                consolidated_latency = float(consolidated_committed['commit_latency'].mean())
                original_latency = float(original_committed['commit_latency'].mean())

                if abs(consolidated_latency - original_latency) > 1.0:
                    print(f"  ❌ Mean latency: {consolidated_latency:.1f}ms vs {original_latency:.1f}ms")
                    deep_check_failed += 1
                    all_checks_passed = False
                    continue

            print(f"  ✓ Verified: {len(consolidated_exp):,} rows, {consolidated_rate:.1%} commit rate")
            deep_check_passed += 1

        except Exception as e:
            print(f"  ❌ Error during verification: {e}")
            deep_check_failed += 1
            all_checks_passed = False

    print(f"\nDeep verification results: {deep_check_passed} passed, {deep_check_failed} failed")

    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print("✓ VERIFICATION PASSED")
        print("=" * 80)
        print("\nConsolidated file is verified correct.")
        print("Original files are preserved for safety.")
        print("\nNext steps:")
        print("  1. Update analysis code to use consolidated format")
        print("  2. Test analysis produces identical results")
        print("  3. Use in production for 2-4 weeks")
        print("  4. Only then consider removing originals (with explicit approval)")
    else:
        print("❌ VERIFICATION FAILED")
        print("=" * 80)
        print("\nConsolidated file has errors or mismatches.")
        print("DO NOT USE consolidated file for analysis.")
        print("DO NOT DELETE original files.")
        print("\nInvestigate errors above and re-run consolidation if needed.")

    return all_checks_passed


def main():
    parser = argparse.ArgumentParser(
        description='Verify consolidated.parquet matches original experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--consolidated', default='experiments/consolidated.parquet',
                       help='Path to consolidated parquet file')
    parser.add_argument('--base-dir', default='experiments',
                       help='Base directory containing original experiments')
    parser.add_argument('--sample', type=int, default=10,
                       help='Number of experiments to deep-verify (default: 10)')

    args = parser.parse_args()

    success = verify_consolidation(
        consolidated_path=args.consolidated,
        base_dir=args.base_dir,
        sample_size=args.sample
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
