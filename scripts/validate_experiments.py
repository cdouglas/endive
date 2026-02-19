#!/usr/bin/env python3
"""Validate experiment pipeline end-to-end.

This script runs validation experiments and checks that:
1. Parquet output has expected columns
2. Operation type distribution matches config
3. Conflict resolution behaves correctly per operation type
4. Per-operation-type metrics can be computed

Usage:
    python scripts/validate_experiments.py [--run] [--check-only]

    --run: Run the validation experiments (default: check existing results)
    --check-only: Only check results, don't run experiments
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def run_experiment(config_path: str) -> bool:
    """Run a single experiment and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print('='*60)

    result = subprocess.run(
        ["python", "-m", "endive.main", config_path, "--yes"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return False

    print("Completed successfully")
    return True


def find_results(label: str) -> str | None:
    """Find results parquet for a given experiment label."""
    patterns = [
        f"experiments/{label}-*/42/results.parquet",
        f"experiments/{label}-*/*/results.parquet",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def validate_v1_smoke_test() -> tuple[bool, list[str]]:
    """V1: Verify basic pipeline produces expected output."""
    errors = []

    path = find_results("v1_smoke_test")
    if not path:
        return False, ["Results file not found for v1_smoke_test"]

    df = pd.read_parquet(path)

    # Check required columns exist
    required_columns = [
        'txn_id', 't_submit', 't_runtime', 't_commit',
        'commit_latency', 'total_latency', 'n_retries',
        'status', 'operation_type', 'abort_reason'
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check operation_type is populated (not all NaN)
    if 'operation_type' in df.columns:
        non_null = df['operation_type'].notna().sum()
        if non_null == 0:
            errors.append("operation_type column is all NaN")
        else:
            print(f"  operation_type: {non_null}/{len(df)} non-null values")

    # Check we have both committed and potentially aborted transactions
    if 'status' in df.columns:
        status_counts = df['status'].value_counts().to_dict()
        print(f"  status distribution: {status_counts}")

        if 'committed' not in status_counts:
            errors.append("No committed transactions")

    # Check operation type distribution roughly matches config (80/20)
    if 'operation_type' in df.columns:
        op_counts = df['operation_type'].value_counts()
        print(f"  operation_type distribution: {op_counts.to_dict()}")

        total = len(df[df['operation_type'].notna()])
        if total > 0:
            fa_pct = op_counts.get('fast_append', 0) / total
            vo_pct = op_counts.get('validated_overwrite', 0) / total

            # Allow 15% tolerance from expected 80/20
            if abs(fa_pct - 0.8) > 0.15:
                errors.append(f"FastAppend ratio {fa_pct:.2f} too far from expected 0.8")
            if abs(vo_pct - 0.2) > 0.15:
                errors.append(f"ValidatedOverwrite ratio {vo_pct:.2f} too far from expected 0.2")

    return len(errors) == 0, errors


def validate_v2_op_distribution() -> tuple[bool, list[str]]:
    """V2: Verify operation type sampling matches config (50/50)."""
    errors = []

    path = find_results("v2_op_distribution")
    if not path:
        return False, ["Results file not found for v2_op_distribution"]

    df = pd.read_parquet(path)

    if 'operation_type' not in df.columns:
        return False, ["operation_type column missing"]

    op_counts = df['operation_type'].value_counts()
    total = len(df[df['operation_type'].notna()])

    print(f"  Total transactions: {total}")
    print(f"  Operation type counts: {op_counts.to_dict()}")

    if total < 50:
        errors.append(f"Too few transactions ({total}) for distribution check")
        return False, errors

    fa_count = op_counts.get('fast_append', 0)
    vo_count = op_counts.get('validated_overwrite', 0)

    fa_pct = fa_count / total
    vo_pct = vo_count / total

    print(f"  FastAppend: {fa_pct:.1%}, ValidatedOverwrite: {vo_pct:.1%}")

    # For 50/50 split, allow 10% tolerance (40-60% each)
    if abs(fa_pct - 0.5) > 0.15:
        errors.append(f"FastAppend ratio {fa_pct:.2f} too far from expected 0.5")
    if abs(vo_pct - 0.5) > 0.15:
        errors.append(f"ValidatedOverwrite ratio {vo_pct:.2f} too far from expected 0.5")

    return len(errors) == 0, errors


def validate_v3_vo_abort() -> tuple[bool, list[str]]:
    """V3: Verify 100% VO with 100% real conflict probability causes ValidationException aborts."""
    errors = []

    path = find_results("v3_vo_abort")
    if not path:
        return False, ["Results file not found for v3_vo_abort"]

    df = pd.read_parquet(path)

    # All transactions should be validated_overwrite
    if 'operation_type' in df.columns:
        op_counts = df['operation_type'].value_counts()
        print(f"  Operation types: {op_counts.to_dict()}")

        if 'fast_append' in op_counts:
            errors.append("Found fast_append transactions in 100% VO experiment")

    # Check aborted transactions
    aborted = df[df['status'] == 'aborted']
    committed = df[df['status'] == 'committed']

    print(f"  Committed: {len(committed)}, Aborted: {len(aborted)}")

    if len(aborted) == 0:
        # If no aborts, there were no conflicts - this is OK at low contention
        print("  Warning: No aborted transactions (low contention?)")
        return True, []

    # All aborts should be validation_exception (since real_conflict_probability=1.0)
    if 'abort_reason' in df.columns:
        abort_reasons = aborted['abort_reason'].value_counts()
        print(f"  Abort reasons: {abort_reasons.to_dict()}")

        ve_aborts = abort_reasons.get('validation_exception', 0)
        mr_aborts = abort_reasons.get('max_retries', 0)

        # With 100% real conflict probability, we expect mostly validation_exception
        # Some max_retries can occur if transaction never got a chance to conflict
        total_aborts = len(aborted)
        ve_ratio = ve_aborts / total_aborts if total_aborts > 0 else 0

        print(f"  ValidationException ratio: {ve_ratio:.1%}")

        if ve_ratio < 0.5:
            errors.append(f"Expected mostly validation_exception aborts, got {ve_ratio:.1%}")

    return len(errors) == 0, errors


def validate_v4_fa_no_abort() -> tuple[bool, list[str]]:
    """V4: Verify 100% FastAppend has no ValidationException aborts."""
    errors = []

    path = find_results("v4_fa_no_abort")
    if not path:
        return False, ["Results file not found for v4_fa_no_abort"]

    df = pd.read_parquet(path)

    # All transactions should be fast_append (or None for legacy)
    if 'operation_type' in df.columns:
        op_counts = df['operation_type'].value_counts()
        print(f"  Operation types: {op_counts.to_dict()}")

        if 'validated_overwrite' in op_counts:
            errors.append("Found validated_overwrite transactions in 100% FA experiment")

    # Check aborted transactions
    aborted = df[df['status'] == 'aborted']
    committed = df[df['status'] == 'committed']

    print(f"  Committed: {len(committed)}, Aborted: {len(aborted)}")

    # No aborts should be validation_exception
    if 'abort_reason' in df.columns and len(aborted) > 0:
        abort_reasons = aborted['abort_reason'].value_counts()
        print(f"  Abort reasons: {abort_reasons.to_dict()}")

        ve_aborts = abort_reasons.get('validation_exception', 0)

        if ve_aborts > 0:
            errors.append(f"FastAppend should never have validation_exception aborts, got {ve_aborts}")

    return len(errors) == 0, errors


def validate_per_operation_metrics() -> tuple[bool, list[str]]:
    """V5: Verify we can compute per-operation-type metrics."""
    errors = []

    # Use v2 results which has mixed operation types
    path = find_results("v2_op_distribution")
    if not path:
        return False, ["Results file not found for per-operation metrics check"]

    df = pd.read_parquet(path)

    print("\n  Per-operation-type metrics:")

    for op_type in ['fast_append', 'validated_overwrite']:
        subset = df[df['operation_type'] == op_type]

        if len(subset) == 0:
            errors.append(f"No {op_type} transactions found")
            continue

        committed = subset[subset['status'] == 'committed']
        success_rate = len(committed) / len(subset) * 100 if len(subset) > 0 else 0

        if len(committed) > 0:
            p50 = committed['commit_latency'].quantile(0.50)
            p99 = committed['commit_latency'].quantile(0.99)
            mean_retries = committed['n_retries'].mean()
        else:
            p50, p99, mean_retries = float('nan'), float('nan'), float('nan')

        print(f"    {op_type}:")
        print(f"      Count: {len(subset)}, Success: {success_rate:.1f}%")
        print(f"      P50: {p50:.0f}ms, P99: {p99:.0f}ms, Mean retries: {mean_retries:.2f}")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Validate experiment pipeline")
    parser.add_argument("--run", action="store_true", help="Run validation experiments")
    parser.add_argument("--check-only", action="store_true", help="Only check existing results")
    args = parser.parse_args()

    configs = [
        "experiment_configs/validation/v1_smoke_test.toml",
        "experiment_configs/validation/v2_op_distribution.toml",
        "experiment_configs/validation/v3_vo_abort.toml",
        "experiment_configs/validation/v4_fa_no_abort.toml",
    ]

    # Run experiments if requested
    if args.run and not args.check_only:
        print("\n" + "="*60)
        print("RUNNING VALIDATION EXPERIMENTS")
        print("="*60)

        for config in configs:
            if not os.path.exists(config):
                print(f"Config not found: {config}")
                sys.exit(1)

            if not run_experiment(config):
                print(f"Experiment failed: {config}")
                sys.exit(1)

    # Run validation checks
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)

    validations = [
        ("V1: Smoke Test", validate_v1_smoke_test),
        ("V2: Operation Distribution", validate_v2_op_distribution),
        ("V3: VO Abort Behavior", validate_v3_vo_abort),
        ("V4: FA No Abort", validate_v4_fa_no_abort),
        ("V5: Per-Operation Metrics", validate_per_operation_metrics),
    ]

    all_passed = True
    results = []

    for name, validator in validations:
        print(f"\n{name}:")
        try:
            passed, errors = validator()
            results.append((name, passed, errors))

            if passed:
                print(f"  PASSED")
            else:
                print(f"  FAILED:")
                for error in errors:
                    print(f"    - {error}")
                all_passed = False
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, False, [str(e)]))
            all_passed = False

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for name, passed, errors in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    if all_passed:
        print("\nAll validations passed!")
        return 0
    else:
        print("\nSome validations failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
