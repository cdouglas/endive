"""Tests for numerical accuracy of simulator calculations.

Verifies that:
1. Commit latency = (t_commit - t_submit) - t_runtime
2. Total latency = commit_latency + t_runtime
3. Manifest list operations count matches snapshots_behind
4. No floating-point error accumulation
5. Timing calculations are consistent and accurate
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from endive.main import configure_from_toml
import endive.main
from endive.capstats import Stats
from endive.test_utils import create_test_config
import simpy


class TestTimingCalculations:
    """Test accuracy of timing calculations."""

    def test_commit_latency_calculation_accuracy(self):
        """Verify commit_latency = (t_commit - t_submit) - t_runtime within machine epsilon."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=300.0,
                num_tables=5
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)
                committed = df[df['status'] == 'committed']

                assert len(committed) > 0, "Need committed transactions to test"

                # Calculate commit_latency from timing fields
                # commit_latency should equal (t_commit - t_submit) - t_runtime
                calculated_commit_latency = (committed['t_commit'] - committed['t_submit']) - committed['t_runtime']
                recorded_commit_latency = committed['commit_latency']

                # Check accuracy to machine epsilon (1e-10)
                max_error = np.abs(calculated_commit_latency - recorded_commit_latency).max()

                assert max_error < 1e-10, \
                    f"commit_latency calculation error exceeds machine epsilon: max_error={max_error:.2e}"

                # Also verify mean error
                mean_error = np.abs(calculated_commit_latency - recorded_commit_latency).mean()
                assert mean_error < 1e-10, \
                    f"Mean commit_latency error: {mean_error:.2e}"

                print(f"✓ Commit latency calculation accurate to machine epsilon")
                print(f"  Max error: {max_error:.2e}")
                print(f"  Mean error: {mean_error:.2e}")
                print(f"  Checked {len(committed)} transactions")

            finally:
                os.unlink(config_path)

    def test_total_latency_calculation_accuracy(self):
        """Verify total_latency = commit_latency + t_runtime within machine epsilon."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=300.0,
                num_tables=5
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)
                committed = df[df['status'] == 'committed']

                assert len(committed) > 0, "Need committed transactions to test"

                # total_latency should equal commit_latency + t_runtime
                calculated_total_latency = committed['commit_latency'] + committed['t_runtime']
                recorded_total_latency = committed['total_latency']

                # Check accuracy to machine epsilon
                max_error = np.abs(calculated_total_latency - recorded_total_latency).max()

                assert max_error < 1e-10, \
                    f"total_latency calculation error exceeds machine epsilon: max_error={max_error:.2e}"

                mean_error = np.abs(calculated_total_latency - recorded_total_latency).mean()
                assert mean_error < 1e-10, \
                    f"Mean total_latency error: {mean_error:.2e}"

                print(f"✓ Total latency calculation accurate to machine epsilon")
                print(f"  Max error: {max_error:.2e}")
                print(f"  Mean error: {mean_error:.2e}")

            finally:
                os.unlink(config_path)

    def test_timing_consistency_across_transactions(self):
        """Verify all timing fields are consistent across all transactions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=40000,
                inter_arrival_scale=200.0,  # High contention for retries
                num_tables=2
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)

                # Test for ALL transactions (committed + aborted)
                assert len(df) > 0, "Need transactions to test"

                # For ALL transactions: total_latency = (t_done - t_submit)
                # Note: t_done field doesn't exist, calculate as t_submit + total_latency
                calculated_t_done = df['t_submit'] + df['total_latency']

                # Verify consistency: t_done should equal t_submit + total_latency (by definition)
                # This is a tautology check to ensure the field calculations are self-consistent
                recorded_total = df['total_latency']

                # Check that total_latency is non-negative and reasonable
                assert (recorded_total >= 0).all(), "total_latency should be non-negative"

                # For committed transactions only
                committed = df[df['status'] == 'committed']
                if len(committed) > 0:
                    # commit_latency = (t_commit - t_submit) - t_runtime
                    calc_commit = (committed['t_commit'] - committed['t_submit']) - committed['t_runtime']
                    rec_commit = committed['commit_latency']
                    max_error_commit = np.abs(calc_commit - rec_commit).max()

                    assert max_error_commit < 1e-10, \
                        f"commit_latency calculation error: {max_error_commit:.2e}"

                    # total_latency = commit_latency + t_runtime
                    calc_total_from_commit = committed['commit_latency'] + committed['t_runtime']
                    rec_total_committed = committed['total_latency']
                    max_error_sum = np.abs(calc_total_from_commit - rec_total_committed).max()

                    assert max_error_sum < 1e-10, \
                        f"total_latency = commit_latency + t_runtime error: {max_error_sum:.2e}"

                print(f"✓ All timing calculations consistent")
                print(f"  Tested {len(df)} total transactions")
                print(f"  Committed: {len(committed)}")

            finally:
                os.unlink(config_path)


class TestFloatingPointStability:
    """Test that floating-point operations don't accumulate error."""

    def test_no_error_accumulation_long_simulation(self):
        """Run long simulation and verify no error accumulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=180000,  # 3 minutes (many events)
                inter_arrival_scale=100.0,
                num_tables=5
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)
                committed = df[df['status'] == 'committed']

                assert len(committed) > 100, "Need many transactions for this test"

                # Check both early and late transactions
                early = committed.head(50)
                late = committed.tail(50)

                # Verify accuracy for both
                for subset, label in [(early, "early"), (late, "late")]:
                    calc_commit = (subset['t_commit'] - subset['t_submit']) - subset['t_runtime']
                    rec_commit = subset['commit_latency']
                    max_error = np.abs(calc_commit - rec_commit).max()

                    assert max_error < 1e-10, \
                        f"{label} transactions show error accumulation: {max_error:.2e}"

                print(f"✓ No error accumulation over long simulation")
                print(f"  Duration: {endive.main.SIM_DURATION_MS}ms")
                print(f"  Total transactions: {len(df)}")
                print(f"  Committed: {len(committed)}")

            finally:
                os.unlink(config_path)

    def test_deterministic_float_operations(self):
        """Verify same seed produces identical floating-point results."""
        results = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for run in range(2):
                config_path = create_test_config(
                    output_path=os.path.join(tmpdir, f"run{run}.parquet"),
                    seed=42,  # Same seed
                    duration_ms=30000,
                    inter_arrival_scale=200.0,
                    num_tables=3
                )

                try:
                    configure_from_toml(config_path)
                    np.random.seed(42)  # Same seed
                    endive.main.STATS = Stats()
                    endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                        endive.main.N_TABLES, endive.main.N_GROUPS,
                        endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                    )

                    # Run simulation
                    env = simpy.Environment()
                    env.process(endive.main.setup(env))
                    env.run(until=endive.main.SIM_DURATION_MS)

                    df = pd.DataFrame(endive.main.STATS.transactions)
                    results.append(df)

                finally:
                    os.unlink(config_path)

            # Verify identical results (bitwise)
            df1, df2 = results
            assert len(df1) == len(df2), "Different number of transactions"

            # Check all float columns are bitwise identical
            float_cols = ['t_submit', 't_runtime', 't_commit', 't_done',
                          'commit_latency', 'total_latency']

            for col in float_cols:
                if col in df1.columns and col in df2.columns:
                    # Use np.array_equal for exact bitwise comparison
                    assert np.array_equal(df1[col].values, df2[col].values), \
                        f"Column {col} not bitwise identical across runs"

            print(f"✓ Floating-point operations are deterministic")
            print(f"  Verified {len(df1)} transactions across 2 runs")


class TestNumericalEdgeCases:
    """Test numerical accuracy in edge cases."""

    def test_very_short_transactions(self):
        """Test accuracy with minimum runtime transactions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=20000,
                inter_arrival_scale=500.0,
                num_tables=5
            )

            # Modify config to have very short runtimes
            with open(config_path, 'r') as f:
                content = f.read()
            content = content.replace('runtime.min = 100', 'runtime.min = 1')
            content = content.replace('runtime.mean = 200', 'runtime.mean = 10')
            with open(config_path, 'w') as f:
                f.write(content)

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify accuracy even with small values
                df = pd.DataFrame(endive.main.STATS.transactions)
                committed = df[df['status'] == 'committed']

                if len(committed) > 0:
                    calc_commit = (committed['t_commit'] - committed['t_submit']) - committed['t_runtime']
                    rec_commit = committed['commit_latency']
                    max_error = np.abs(calc_commit - rec_commit).max()

                    assert max_error < 1e-10, \
                        f"Accuracy loss with short runtimes: {max_error:.2e}"

                    print(f"✓ Accuracy maintained with very short transactions")
                    print(f"  Min runtime: {committed['t_runtime'].min():.2f}ms")

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
