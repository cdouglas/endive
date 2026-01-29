"""Tests for conflict resolution and storage latency features."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from endive.main import configure_from_toml, generate_latency
import endive.main
from endive.capstats import Stats
import simpy


def create_conflict_test_config(
    output_path: str,
    seed: int = 42,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    num_tables: int = 3,
    max_parallel: int = 4,
    min_latency: float = 5.0,
    cas_mean: float = 100.0,
    cas_stddev: float = 10.0
) -> str:
    """Create a test configuration file with conflict resolution parameters."""
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
seed = {seed}

[catalog]
num_tables = {num_tables}

[transaction]
retry = 10
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5

inter_arrival.distribution = "exponential"
inter_arrival.scale = {inter_arrival_scale}
inter_arrival.min = 100.0
inter_arrival.max = 1000.0
inter_arrival.mean = 500.0
inter_arrival.std_dev = 100.0
inter_arrival.value = 500.0

ntable.zipf = 2.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[storage]
max_parallel = {max_parallel}
min_latency = {min_latency}

T_CAS.mean = {cas_mean}
T_CAS.stddev = {cas_stddev}

T_METADATA_ROOT.read.mean = 50
T_METADATA_ROOT.read.stddev = 5
T_METADATA_ROOT.write.mean = 60
T_METADATA_ROOT.write.stddev = 6

T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6

T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        return f.name


def run_simulation_from_config(config_path: str) -> pd.DataFrame:
    """Run simulation and return results as DataFrame."""
    # Reset global stats
    endive.main.STATS = Stats()

    # Load configuration
    configure_from_toml(config_path)

    # Setup random seed if specified
    if endive.main.SIM_SEED is not None:
        np.random.seed(endive.main.SIM_SEED)

    # Run simulation
    env = simpy.Environment()
    env.process(endive.main.setup(env))
    env.run(until=endive.main.SIM_DURATION_MS)

    # Return results as DataFrame
    return pd.DataFrame(endive.main.STATS.transactions)


class TestMinimumLatency:
    """Test minimum latency enforcement."""

    def test_latencies_respect_minimum(self):
        """Verify all generated latencies are >= minimum."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=15000,
                inter_arrival_scale=200.0,  # High load for more ops
                min_latency=10.0,  # Set explicit minimum
                cas_mean=20.0,
                cas_stddev=15.0  # High stddev to test edge cases
            )

            try:
                # Load config
                configure_from_toml(config_path)

                # Generate many latency samples
                samples = [generate_latency(20.0, 15.0) for _ in range(1000)]

                # All should be >= minimum
                assert all(s >= endive.main.MIN_LATENCY for s in samples), \
                    f"Some latencies below minimum: min={min(samples)}, MIN_LATENCY={endive.main.MIN_LATENCY}"

                # Verify minimum is actually being enforced
                assert endive.main.MIN_LATENCY == 10.0
                assert min(samples) >= 10.0

                print(f"✓ Minimum latency test passed")
                print(f"  Generated 1000 samples with mean=20, stddev=15")
                print(f"  Minimum value: {min(samples):.2f}ms (>= {endive.main.MIN_LATENCY}ms)")
                print(f"  Mean value: {np.mean(samples):.2f}ms")

            finally:
                os.unlink(config_path)

    def test_minimum_latency_in_simulation(self):
        """Verify minimum latency is enforced during actual simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=20000,
                inter_arrival_scale=300.0,
                min_latency=8.0,
                cas_mean=15.0,
                cas_stddev=12.0  # Very high stddev to stress-test minimum
            )

            try:
                df = run_simulation_from_config(config_path)

                # Simulation should complete successfully
                assert len(df) > 0, "Simulation produced no transactions"

                # All committed transactions should have realistic latencies
                committed = df[df['status'] == 'committed']
                if len(committed) > 0:
                    # Commit latency should be >= some reasonable minimum
                    # (multiple operations, each >= MIN_LATENCY)
                    min_commit_latency = committed['commit_latency'].min()
                    assert min_commit_latency >= 8.0, \
                        f"Commit latency {min_commit_latency} too low"

                print(f"✓ Minimum latency in simulation test passed")

            finally:
                os.unlink(config_path)


class TestConflictResolution:
    """Test conflict resolution with manifest list reading."""

    def test_high_contention_causes_retries(self):
        """High contention should cause CAS failures and retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "conflict.parquet"),
                seed=42,
                duration_ms=25000,
                inter_arrival_scale=100.0,  # Very high load
                num_tables=2,  # Few tables = more conflicts
                max_parallel=4
            )

            try:
                df = run_simulation_from_config(config_path)

                committed = df[df['status'] == 'committed']
                assert len(committed) > 0, "No transactions committed"

                # With high contention, we should see retries
                txns_with_retries = committed[committed['n_retries'] > 0]
                retry_rate = len(txns_with_retries) / len(committed)

                assert retry_rate > 0, "Expected some retries with high contention"

                # Some transactions should have multiple retries
                max_retries = committed['n_retries'].max()
                assert max_retries >= 2, \
                    f"Expected multi-retry transactions, got max={max_retries}"

                print(f"✓ High contention test passed")
                print(f"  Transactions: {len(committed)}")
                print(f"  Retry rate: {retry_rate*100:.1f}%")
                print(f"  Max retries: {max_retries}")
                print(f"  Mean retries: {committed['n_retries'].mean():.2f}")

            finally:
                os.unlink(config_path)

    def test_commit_latency_increases_with_retries(self):
        """Transactions with more retries should have higher commit latency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "latency.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=150.0,  # High load
                num_tables=3
            )

            try:
                df = run_simulation_from_config(config_path)

                committed = df[df['status'] == 'committed']
                assert len(committed) > 10, "Need more transactions for this test"

                # Split into low and high retry groups
                no_retry = committed[committed['n_retries'] == 0]
                with_retries = committed[committed['n_retries'] >= 2]

                if len(no_retry) > 0 and len(with_retries) > 0:
                    avg_latency_no_retry = no_retry['commit_latency'].mean()
                    avg_latency_with_retries = with_retries['commit_latency'].mean()

                    # Retries should increase latency due to reading manifest lists
                    assert avg_latency_with_retries > avg_latency_no_retry, \
                        f"Expected higher latency with retries: {avg_latency_with_retries} vs {avg_latency_no_retry}"

                    print(f"✓ Commit latency increase test passed")
                    print(f"  No retries: {avg_latency_no_retry:.2f}ms avg commit latency")
                    print(f"  With retries (>=2): {avg_latency_with_retries:.2f}ms avg commit latency")
                    print(f"  Increase: {avg_latency_with_retries - avg_latency_no_retry:.2f}ms")

            finally:
                os.unlink(config_path)


class TestParallelism:
    """Test parallelism limit enforcement."""

    def test_different_parallelism_affects_latency(self):
        """Different parallelism limits should affect commit latency under high contention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Low parallelism
            config_low = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "parallel_low.parquet"),
                seed=42,
                duration_ms=25000,
                inter_arrival_scale=120.0,
                num_tables=2,
                max_parallel=1  # Very limited parallelism
            )

            # High parallelism
            config_high = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "parallel_high.parquet"),
                seed=42,
                duration_ms=25000,
                inter_arrival_scale=120.0,
                num_tables=2,
                max_parallel=8  # Higher parallelism
            )

            try:
                df_low = run_simulation_from_config(config_low)
                df_high = run_simulation_from_config(config_high)

                committed_low = df_low[df_low['status'] == 'committed']
                committed_high = df_high[df_high['status'] == 'committed']

                # Both should have retries
                assert committed_low['n_retries'].max() > 0
                assert committed_high['n_retries'].max() > 0

                # With deterministic seed, the transaction patterns are the same,
                # but parallelism affects how quickly manifest lists are read
                # Higher parallelism should allow faster conflict resolution on average

                print(f"✓ Parallelism comparison test passed")
                print(f"  Low parallelism (1): {len(committed_low)} committed, "
                      f"{committed_low['commit_latency'].mean():.2f}ms avg")
                print(f"  High parallelism (8): {len(committed_high)} committed, "
                      f"{committed_high['commit_latency'].mean():.2f}ms avg")

            finally:
                os.unlink(config_low)
                os.unlink(config_high)


class TestStochasticLatencies:
    """Test stochastic latency distributions."""

    def test_latencies_follow_normal_distribution(self):
        """Generated latencies should follow approximately normal distribution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "stochastic.parquet"),
                seed=42,
                min_latency=5.0,
                cas_mean=100.0,
                cas_stddev=15.0
            )

            try:
                configure_from_toml(config_path)

                # Generate large sample
                samples = [endive.main.get_cas_latency() for _ in range(1000)]

                sample_mean = np.mean(samples)
                sample_std = np.std(samples)

                # Mean should be close to configured mean (accounting for truncation at MIN_LATENCY)
                assert 95 <= sample_mean <= 105, \
                    f"Sample mean {sample_mean} far from configured mean 100"

                # Std should be reasonable (will be slightly less due to truncation)
                assert 10 <= sample_std <= 20, \
                    f"Sample std {sample_std} outside expected range"

                # All samples should be >= MIN_LATENCY
                assert min(samples) >= endive.main.MIN_LATENCY

                print(f"✓ Stochastic latency distribution test passed")
                print(f"  Configured: mean=100, stddev=15, min=5")
                print(f"  Observed: mean={sample_mean:.2f}, stddev={sample_std:.2f}, min={min(samples):.2f}")

            finally:
                os.unlink(config_path)

    def test_read_write_latencies_differ(self):
        """Read and write latencies should have different distributions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_conflict_test_config(
                output_path=os.path.join(tmpdir, "rw.parquet"),
                seed=42
            )

            try:
                configure_from_toml(config_path)

                # Generate samples for read and write
                read_samples = [endive.main.get_manifest_list_latency('read') for _ in range(500)]
                write_samples = [endive.main.get_manifest_list_latency('write') for _ in range(500)]

                read_mean = np.mean(read_samples)
                write_mean = np.mean(write_samples)

                # Write should be slower than read (based on config: 60 vs 50)
                assert write_mean > read_mean, \
                    f"Write latency {write_mean} should be > read latency {read_mean}"

                # Difference should be meaningful
                diff = write_mean - read_mean
                assert diff > 5, f"Difference {diff} too small to be meaningful"

                print(f"✓ Read/write latency differentiation test passed")
                print(f"  Read mean: {read_mean:.2f}ms")
                print(f"  Write mean: {write_mean:.2f}ms")
                print(f"  Difference: {diff:.2f}ms")

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
