"""Regression tests for saturation analysis.

These tests verify measurement accuracy and would have detected the throughput
calculation bug (using max instead of max-min after warmup/cooldown filtering).

Test Philosophy:
- Verify that analysis measures what it claims to measure
- Catch regressions in duration/throughput calculations
- Ensure warmup/cooldown filtering doesn't corrupt statistics
- Test with realistic simulation timescales (1 hour)
"""

import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from endive.saturation_analysis import (
    load_and_aggregate_results,
    compute_aggregate_statistics,
    compute_per_seed_statistics,
    compute_transient_period_duration,
    build_experiment_index
)
import endive.saturation_analysis as saturation_analysis


class TestThroughputCalculationAccuracy:
    """Test that throughput is calculated correctly after warmup/cooldown filtering.

    This test class would have caught the regression where duration was calculated
    as max(t_submit) instead of (max(t_submit) - min(t_submit)) after filtering.
    """

    def test_throughput_with_warmup_cooldown_filtering(self):
        """Verify throughput calculation accounts for filtered timestamps.

        REGRESSION TEST: This catches the bug where duration was calculated as
        max(t_submit)/1000 instead of (max-min)/1000 after warmup/cooldown filtering.

        Setup: 1-hour simulation with known throughput
        Expected: Measured throughput matches known rate in steady-state window
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dir = Path(tmpdir) / "12345"
            seed_dir.mkdir()

            # Simulation: 1 hour (3600 seconds)
            # Mean transaction runtime: 180 seconds
            # K_MIN_CYCLES: 5
            # Warmup: 5 * 180s = 900s = 15 minutes
            # Cooldown: 900s = 15 minutes
            # Active window: 3600 - 900 - 900 = 1800s = 30 minutes

            # Known rate: 1 transaction every 100 seconds in steady state
            # Expected: 1800s / 100s = 18 transactions
            # Expected throughput: 18 / 1800s = 0.01 commits/sec

            transaction_interval_ms = 100000  # 100 seconds

            # Generate transactions throughout entire hour
            # But only those in [15min, 45min] window should be counted
            txn_data = []
            txn_id = 0

            # Transactions before warmup (excluded)
            for i in range(3):
                t_submit = i * transaction_interval_ms
                txn_data.append({
                    'txn_id': txn_id,
                    't_submit': t_submit,
                    't_commit': t_submit + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })
                txn_id += 1

            # Transactions in active window (15min to 45min)
            # 15 minutes = 900000ms, 45 minutes = 2700000ms
            for i in range(18):
                t_submit = 900000 + i * transaction_interval_ms
                txn_data.append({
                    'txn_id': txn_id,
                    't_submit': t_submit,
                    't_commit': t_submit + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })
                txn_id += 1

            # Transactions after cooldown (excluded)
            for i in range(3):
                t_submit = 2700000 + i * transaction_interval_ms
                txn_data.append({
                    'txn_id': txn_id,
                    't_submit': t_submit,
                    't_commit': t_submit + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })
                txn_id += 1

            df = pd.DataFrame(txn_data)
            df.to_parquet(seed_dir / "results.parquet")

            # Load with warmup/cooldown filtering
            exp_info = {
                'seeds': [str(seed_dir)],
                'config': {
                    'transaction': {'runtime': {'mean': 180000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }

            # Save original CONFIG
            original_config = saturation_analysis.CONFIG.copy()
            try:
                # Set K_MIN_CYCLES = 5 (default)
                saturation_analysis.CONFIG = {
                    'analysis': {
                        'k_min_cycles': 5,
                        'min_warmup_ms': 300000,
                        'max_warmup_ms': 900000,
                        'min_seeds': 1
                    }
                }

                result_df = load_and_aggregate_results(exp_info)
                stats = compute_aggregate_statistics(result_df)

                # After filtering, should have 18 transactions in active window
                assert stats['committed'] == 18, \
                    f"Expected 18 committed txns in active window, got {stats['committed']}"

                # Throughput should be 18 commits / 1800s = 0.01 commits/sec
                expected_throughput = 18 / 1800.0
                measured_throughput = stats['throughput']

                # Allow 10% tolerance (edge effects from discrete transactions near boundaries)
                assert abs(measured_throughput - expected_throughput) / expected_throughput < 0.10, \
                    f"Expected throughput {expected_throughput:.4f} c/s, got {measured_throughput:.4f} c/s. " \
                    f"This suggests duration calculation is wrong (using max instead of max-min)."

                print(f"✓ Throughput correctly calculated: {measured_throughput:.4f} c/s (expected {expected_throughput:.4f} c/s)")
            finally:
                saturation_analysis.CONFIG = original_config

    def test_duration_calculation_with_offset_timestamps(self):
        """Verify duration uses (max-min) not just max for offset timestamps.

        REGRESSION TEST: Directly tests the bug fix.

        After warmup filtering, timestamps don't start at 0. Duration must be
        calculated as (max - min), not just max.
        """
        # Create dataframe with timestamps offset from 0 (like after filtering)
        df = pd.DataFrame({
            't_submit': [900000, 1000000, 1100000, 1200000],  # Start at 15 minutes
            'commit_latency': [1000] * 4,
            'total_latency': [11000] * 4,
            'n_retries': [0] * 4,
            'status': ['committed'] * 4
        })

        stats = compute_aggregate_statistics(df)

        # Duration should be (1200000 - 900000) / 1000 = 300 seconds
        # NOT 1200000 / 1000 = 1200 seconds (the bug)

        expected_duration_s = (1200000 - 900000) / 1000.0  # 300s
        expected_throughput = 4 / expected_duration_s  # 4 commits / 300s = 0.0133 c/s

        measured_throughput = stats['throughput']

        # If the bug exists, throughput would be 4/1200 = 0.00333 c/s (4× too low)
        assert abs(measured_throughput - expected_throughput) < 0.001, \
            f"Expected throughput {expected_throughput:.4f} c/s, got {measured_throughput:.4f} c/s. " \
            f"Duration likely calculated as max(t_submit) instead of (max-min)."

        print(f"✓ Duration correctly calculated from offset timestamps: {measured_throughput:.4f} c/s")

    def test_per_seed_throughput_with_filtering(self):
        """Verify per-seed statistics also use correct duration calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dirs = []

            for seed in [111, 222]:
                seed_dir = Path(tmpdir) / str(seed)
                seed_dir.mkdir()
                seed_dirs.append(str(seed_dir))

                # Create data spanning 1 hour with known rate
                # 10 transactions in active window [15min, 45min]
                txn_data = []

                # Active window: 900000ms to 2700000ms (30 minutes = 1800s)
                for i in range(10):
                    t_submit = 900000 + i * 180000  # Every 3 minutes
                    txn_data.append({
                        'txn_id': i,
                        't_submit': t_submit,
                        't_commit': t_submit + 1000,
                        'commit_latency': 1000,
                        'total_latency': 11000,
                        'n_retries': 0,
                        'status': 'committed'
                    })

                df = pd.DataFrame(txn_data)
                df.to_parquet(seed_dir / "results.parquet")

            # Load with warmup/cooldown filtering
            exp_info = {
                'seeds': seed_dirs,
                'config': {
                    'transaction': {'runtime': {'mean': 180000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }

            original_config = saturation_analysis.CONFIG.copy()
            try:
                saturation_analysis.CONFIG = {
                    'analysis': {
                        'k_min_cycles': 5,
                        'min_warmup_ms': 300000,
                        'max_warmup_ms': 900000,
                        'min_seeds': 1
                    }
                }

                result_df = load_and_aggregate_results(exp_info)
                per_seed_df = compute_per_seed_statistics(result_df)

                # Each seed should have 10 commits over 1800s
                expected_throughput = 10 / 1800.0

                for _, row in per_seed_df.iterrows():
                    assert abs(row['throughput'] - expected_throughput) < 0.001, \
                        f"Seed {row['seed']}: Expected throughput {expected_throughput:.4f} c/s, " \
                        f"got {row['throughput']:.4f} c/s"

                print(f"✓ Per-seed throughput correctly calculated for {len(per_seed_df)} seeds")
            finally:
                saturation_analysis.CONFIG = original_config

    def test_throughput_independent_of_k_min_cycles(self):
        """Verify measured throughput is same regardless of K_MIN_CYCLES value.

        If we measure a smaller or larger steady-state window, the throughput
        (commits per second) should be the same as long as system is in steady state.

        This is a fundamental sanity check: throughput is an intensive property.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dir = Path(tmpdir) / "12345"
            seed_dir.mkdir()

            # Generate transactions at constant rate throughout simulation
            # Rate: 1 transaction every 60 seconds for 1 hour
            txn_data = []
            for i in range(60):
                t_submit = i * 60000
                txn_data.append({
                    'txn_id': i,
                    't_submit': t_submit,
                    't_commit': t_submit + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })

            df = pd.DataFrame(txn_data)
            df.to_parquet(seed_dir / "results.parquet")

            exp_info = {
                'seeds': [str(seed_dir)],
                'config': {
                    'transaction': {'runtime': {'mean': 180000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }

            original_config = saturation_analysis.CONFIG.copy()
            throughputs = {}

            try:
                # Test with different K_MIN_CYCLES values
                for k in [3, 5, 7]:
                    saturation_analysis.CONFIG = {
                        'analysis': {
                            'k_min_cycles': k,
                            'min_warmup_ms': 300000,
                            'max_warmup_ms': 900000,
                            'min_seeds': 1
                        }
                    }

                    result_df = load_and_aggregate_results(exp_info)
                    stats = compute_aggregate_statistics(result_df)
                    throughputs[k] = stats['throughput']

                # All throughputs should be approximately equal (within 5%)
                # Small differences expected due to edge effects
                throughput_values = list(throughputs.values())
                mean_throughput = np.mean(throughput_values)

                for k, tp in throughputs.items():
                    rel_diff = abs(tp - mean_throughput) / mean_throughput
                    assert rel_diff < 0.05, \
                        f"K={k}: Throughput {tp:.4f} differs by {rel_diff*100:.1f}% from mean {mean_throughput:.4f}. " \
                        f"Throughput should be independent of measurement window size in steady state."

                print(f"✓ Throughput consistent across K values: {throughputs}")
            finally:
                saturation_analysis.CONFIG = original_config


class TestWarmupCooldownFiltering:
    """Test that warmup/cooldown filtering works correctly."""

    def test_warmup_computation_uses_config(self):
        """Verify warmup duration respects CONFIG values."""
        original_config = saturation_analysis.CONFIG.copy()

        try:
            # Test with custom K_MIN_CYCLES
            saturation_analysis.CONFIG = {
                'analysis': {
                    'k_min_cycles': 7,
                    'min_warmup_ms': 300000,
                    'max_warmup_ms': 900000
                }
            }

            config = {
                'transaction': {
                    'runtime': {
                        'mean': 60000  # 60 seconds
                    }
                }
            }

            warmup = compute_transient_period_duration(config)

            # Expected: 7 * 60s = 420s = 420000ms
            # Clamped to [300000, 900000]
            expected = 420000
            assert warmup == expected, \
                f"Expected warmup {expected}ms with K=7 and mean_runtime=60s, got {warmup}ms"

            print(f"✓ Warmup computation uses CONFIG correctly: {warmup}ms")
        finally:
            saturation_analysis.CONFIG = original_config

    def test_transactions_outside_window_excluded(self):
        """Verify transactions outside active window are excluded from statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dir = Path(tmpdir) / "12345"
            seed_dir.mkdir()

            # Create transactions throughout 1-hour simulation
            # With K=5 and mean_runtime=180s, active window is [900s, 2700s]
            txn_data = []

            # 5 transactions in warmup period (excluded)
            for i in range(5):
                txn_data.append({
                    'txn_id': i,
                    't_submit': i * 100000,  # 0s, 100s, 200s, 300s, 400s
                    't_commit': i * 100000 + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })

            # 10 transactions in active window (included)
            for i in range(10):
                txn_data.append({
                    'txn_id': 100 + i,
                    't_submit': 1000000 + i * 100000,  # 1000s to 1900s
                    't_commit': 1000000 + i * 100000 + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })

            # 5 transactions in cooldown period (excluded)
            for i in range(5):
                txn_data.append({
                    'txn_id': 200 + i,
                    't_submit': 2800000 + i * 100000,  # 2800s to 3200s
                    't_commit': 2800000 + i * 100000 + 1000,
                    'commit_latency': 1000,
                    'total_latency': 11000,
                    'n_retries': 0,
                    'status': 'committed'
                })

            df = pd.DataFrame(txn_data)
            df.to_parquet(seed_dir / "results.parquet")

            exp_info = {
                'seeds': [str(seed_dir)],
                'config': {
                    'transaction': {'runtime': {'mean': 180000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }

            original_config = saturation_analysis.CONFIG.copy()
            try:
                saturation_analysis.CONFIG = {
                    'analysis': {
                        'k_min_cycles': 5,
                        'min_warmup_ms': 300000,
                        'max_warmup_ms': 900000,
                        'min_seeds': 1
                    }
                }

                result_df = load_and_aggregate_results(exp_info)
                stats = compute_aggregate_statistics(result_df)

                # Should only count the 10 transactions in active window
                assert stats['total_txns'] == 10, \
                    f"Expected 10 txns in active window, got {stats['total_txns']}"
                assert stats['committed'] == 10, \
                    f"Expected 10 committed in active window, got {stats['committed']}"

                print(f"✓ Correctly excluded {20-10} transactions outside active window")
            finally:
                saturation_analysis.CONFIG = original_config


class TestStatisticalAccuracy:
    """Test that statistics accurately reflect the filtered data."""

    def test_overhead_percentage_calculation(self):
        """Verify overhead percentage is calculated correctly.

        Overhead = (commit_latency / total_latency) * 100
        """
        df = pd.DataFrame({
            't_submit': [0, 1000, 2000, 3000],
            'commit_latency': [100, 200, 300, 400],  # Time in commit protocol
            'total_latency': [1000, 2000, 3000, 4000],  # Total transaction time
            'n_retries': [0, 1, 2, 3],
            'status': ['committed'] * 4
        })

        stats = compute_aggregate_statistics(df)

        # Each transaction has 10% overhead (commit_latency / total_latency)
        expected_overhead = 10.0

        assert abs(stats['mean_overhead_pct'] - expected_overhead) < 0.1, \
            f"Expected mean overhead {expected_overhead}%, got {stats['mean_overhead_pct']:.1f}%"

        print(f"✓ Overhead percentage correctly calculated: {stats['mean_overhead_pct']:.1f}%")

    def test_success_rate_with_mixed_status(self):
        """Verify success rate calculation with failures."""
        df = pd.DataFrame({
            't_submit': range(10),
            'commit_latency': [1000] * 10,
            'total_latency': [11000] * 10,
            'n_retries': [0] * 10,
            'status': ['committed'] * 7 + ['aborted'] * 3
        })

        stats = compute_aggregate_statistics(df)

        # 7 out of 10 succeeded
        expected_success_rate = 70.0

        assert stats['success_rate'] == expected_success_rate, \
            f"Expected success rate {expected_success_rate}%, got {stats['success_rate']:.1f}%"
        assert stats['committed'] == 7
        assert stats['aborted'] == 3

        print(f"✓ Success rate correctly calculated: {stats['success_rate']:.1f}%")

    def test_retry_statistics_only_for_committed(self):
        """Verify mean_retries computed over committed transactions only."""
        df = pd.DataFrame({
            't_submit': range(10),
            'commit_latency': [1000] * 10,
            'total_latency': [11000] * 10,
            'n_retries': [0, 1, 2, 3, 4] + [10, 10, 10, 10, 10],  # High retries for aborted
            'status': ['committed'] * 5 + ['aborted'] * 5
        })

        stats = compute_aggregate_statistics(df)

        # Mean retries should be over committed only: (0+1+2+3+4)/5 = 2.0
        # NOT over all transactions: (0+1+2+3+4+10+10+10+10+10)/10 = 6.0
        expected_mean_retries = 2.0

        assert abs(stats['mean_retries'] - expected_mean_retries) < 0.01, \
            f"Expected mean retries {expected_mean_retries} (over committed only), " \
            f"got {stats['mean_retries']:.2f}"

        print(f"✓ Mean retries correctly computed over committed only: {stats['mean_retries']:.1f}")


class TestEndToEndAccuracy:
    """End-to-end tests verifying complete analysis accuracy."""

    def test_full_experiment_analysis_accuracy(self):
        """Verify complete experiment analysis produces accurate results.

        This is an integration test that exercises the full pipeline with
        realistic data and verifies all computed statistics are accurate.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment with known properties
            exp_dir = Path(tmpdir) / "accuracy_test-abc123"
            exp_dir.mkdir()

            # Config
            cfg_content = """
[simulation]
duration_ms = 3600000

[experiment]
label = "accuracy_test"

[catalog]
num_tables = 1
num_groups = 1

[transaction]
inter_arrival.scale = 100
real_conflict_probability = 0.5
runtime.mean = 180000
"""
            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write(cfg_content)

            # Create 3 seeds with identical steady-state behavior
            for seed in [111, 222, 333]:
                seed_dir = exp_dir / str(seed)
                seed_dir.mkdir()

                # Generate transactions at constant rate in active window
                # Active window: [900s, 2700s] = 1800s
                # Rate: 1 transaction per 100s
                # Expected: 18 transactions
                txn_data = []

                for i in range(18):
                    t_submit = 900000 + i * 100000
                    txn_data.append({
                        'txn_id': i,
                        't_submit': t_submit,
                        't_commit': t_submit + 5000,
                        'commit_latency': 2000,  # 2 seconds in commit
                        'total_latency': 5000,   # 5 seconds total
                        'n_retries': i % 3,  # 0, 1, or 2 retries
                        'status': 'committed' if i < 15 else 'aborted',  # 15/18 succeed
                        't_runtime': 3000  # 3 seconds transaction execution
                    })

                df = pd.DataFrame(txn_data)
                df.to_parquet(seed_dir / "results.parquet")

            # Build index
            original_config = saturation_analysis.CONFIG.copy()
            try:
                saturation_analysis.CONFIG = {
                    'analysis': {
                        'k_min_cycles': 5,
                        'min_warmup_ms': 300000,
                        'max_warmup_ms': 900000,
                        'min_seeds': 1
                    }
                }

                index_df = build_experiment_index(tmpdir, "accuracy_test-*")

                assert len(index_df) == 1
                row = index_df.iloc[0]

                # Verify all statistics are accurate
                assert row['num_seeds'] == 3
                assert row['total_txns'] == 18 * 3  # 54 total
                assert row['committed'] == 15 * 3  # 45 committed

                expected_success_rate = (15.0 / 18.0) * 100  # 83.33%
                assert abs(row['success_rate'] - expected_success_rate) < 0.1

                expected_throughput = 15 / 1800.0  # 15 commits per 1800s per seed
                assert abs(row['throughput'] - expected_throughput) < 0.001, \
                    f"Expected throughput {expected_throughput:.4f} c/s, got {row['throughput']:.4f} c/s"

                # Verify latencies
                assert row['p50_commit_latency'] == 2000  # All have 2000ms commit latency

                # Verify overhead (commit_latency / total_latency)
                expected_overhead = (2000.0 / 5000.0) * 100  # 40%
                assert abs(row['mean_overhead_pct'] - expected_overhead) < 0.1

                print(f"✓ Full experiment analysis produces accurate results:")
                print(f"  - Success rate: {row['success_rate']:.1f}%")
                print(f"  - Throughput: {row['throughput']:.4f} c/s")
                print(f"  - Overhead: {row['mean_overhead_pct']:.1f}%")
            finally:
                saturation_analysis.CONFIG = original_config


class TestRegressionDetection:
    """Meta-tests that verify these tests would catch known regressions."""

    def test_would_catch_max_instead_of_max_minus_min_bug(self):
        """Verify our tests would catch the original throughput bug.

        Simulate the bug by manually calculating throughput the wrong way,
        and verify our test assertions would fail.
        """
        # Create data with offset timestamps (like after filtering)
        df = pd.DataFrame({
            't_submit': [900000, 1000000, 1100000, 1200000],
            'commit_latency': [1000] * 4,
            'total_latency': [11000] * 4,
            'n_retries': [0] * 4,
            'status': ['committed'] * 4
        })

        # Correct calculation
        correct_duration = (df['t_submit'].max() - df['t_submit'].min()) / 1000.0
        correct_throughput = len(df) / correct_duration

        # Buggy calculation (what the code was doing)
        buggy_duration = df['t_submit'].max() / 1000.0
        buggy_throughput = len(df) / buggy_duration

        # Verify there's a significant difference
        ratio = buggy_throughput / correct_throughput
        assert ratio < 0.5, \
            f"Bug simulation failed: buggy throughput should be much lower than correct"

        # Verify our test would catch this
        expected_throughput = 4 / 300.0  # 4 commits in 300 seconds

        # Test with correct value would pass
        assert abs(correct_throughput - expected_throughput) < 0.001

        # Test with buggy value would fail
        try:
            assert abs(buggy_throughput - expected_throughput) < 0.001
            pytest.fail("Test should have failed with buggy calculation, but didn't!")
        except AssertionError:
            # Expected - the test correctly detected the bug
            pass

        print(f"✓ Tests would catch max-instead-of-max-minus-min bug")
        print(f"  - Correct throughput: {correct_throughput:.4f} c/s")
        print(f"  - Buggy throughput: {buggy_throughput:.4f} c/s (off by {ratio:.2f}x)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
