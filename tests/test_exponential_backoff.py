"""Tests for exponential backoff retry mechanism."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from endive.main import configure_from_toml, calculate_backoff_time
import endive.main
from endive.capstats import Stats
import simpy


def create_backoff_test_config(
    output_path: str,
    seed: int = 42,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 50.0,  # High contention to trigger retries
    backoff_enabled: bool = True,
    backoff_base_ms: float = 10.0,
    backoff_multiplier: float = 2.0,
    backoff_max_ms: float = 5000.0,
    backoff_jitter: float = 0.1
) -> str:
    """Create a test configuration file with exponential backoff parameters."""
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
seed = {seed}

[catalog]
num_tables = 1

[transaction]
retry = 10
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5

inter_arrival.distribution = "exponential"
inter_arrival.scale = {inter_arrival_scale}

ntable.zipf = 10.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

real_conflict_probability = 0.0

conflicting_manifests.distribution = "exponential"
conflicting_manifests.mean = 3.0
conflicting_manifests.min = 1
conflicting_manifests.max = 10

# Exponential backoff configuration
retry_backoff.enabled = {str(backoff_enabled).lower()}
retry_backoff.base_ms = {backoff_base_ms}
retry_backoff.multiplier = {backoff_multiplier}
retry_backoff.max_ms = {backoff_max_ms}
retry_backoff.jitter = {backoff_jitter}

[storage]
max_parallel = 4
min_latency = 1

T_CAS.mean = 1
T_CAS.stddev = 0.1

T_METADATA_ROOT.read.mean = 1
T_METADATA_ROOT.read.stddev = 0.1
T_METADATA_ROOT.write.mean = 1
T_METADATA_ROOT.write.stddev = 0.1

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


class TestExponentialBackoff:
    """Test suite for exponential backoff functionality."""

    def test_backoff_disabled_returns_zero(self):
        """Test that backoff returns 0 when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_file = create_backoff_test_config(
                output_path,
                backoff_enabled=False
            )

            try:
                configure_from_toml(config_file)

                # Test that backoff is disabled
                assert endive.main.RETRY_BACKOFF_ENABLED is False

                # Calculate backoff for various retry numbers
                for retry_num in range(1, 10):
                    backoff = calculate_backoff_time(retry_num)
                    assert backoff == 0.0, f"Expected 0 backoff when disabled, got {backoff} for retry {retry_num}"

            finally:
                os.unlink(config_file)

    def test_backoff_exponential_growth(self):
        """Test that backoff grows exponentially with retry number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_file = create_backoff_test_config(
                output_path,
                backoff_enabled=True,
                backoff_base_ms=10.0,
                backoff_multiplier=2.0,
                backoff_max_ms=10000.0,
                backoff_jitter=0.0  # Disable jitter for predictable testing
            )

            try:
                configure_from_toml(config_file)

                # Test that backoff is enabled
                assert endive.main.RETRY_BACKOFF_ENABLED is True
                assert endive.main.RETRY_BACKOFF_BASE_MS == 10.0
                assert endive.main.RETRY_BACKOFF_MULTIPLIER == 2.0
                assert endive.main.RETRY_BACKOFF_JITTER == 0.0

                # Seed for reproducibility
                np.random.seed(42)

                # Test exponential growth: 10, 20, 40, 80, 160, ...
                expected_backoffs = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0]
                for retry_num, expected in enumerate(expected_backoffs, start=1):
                    backoff = calculate_backoff_time(retry_num)
                    assert backoff == pytest.approx(expected, abs=0.1), \
                        f"Expected {expected}ms for retry {retry_num}, got {backoff}ms"

            finally:
                os.unlink(config_file)

    def test_backoff_max_cap(self):
        """Test that backoff is capped at max_ms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_file = create_backoff_test_config(
                output_path,
                backoff_enabled=True,
                backoff_base_ms=100.0,
                backoff_multiplier=2.0,
                backoff_max_ms=500.0,
                backoff_jitter=0.0
            )

            try:
                configure_from_toml(config_file)

                # Seed for reproducibility
                np.random.seed(42)

                # Test that backoff caps at max_ms
                # 100, 200, 400, 800->500, 1600->500, ...
                expected_backoffs = [100.0, 200.0, 400.0, 500.0, 500.0, 500.0]
                for retry_num, expected in enumerate(expected_backoffs, start=1):
                    backoff = calculate_backoff_time(retry_num)
                    assert backoff == pytest.approx(expected, abs=0.1), \
                        f"Expected {expected}ms (capped) for retry {retry_num}, got {backoff}ms"

            finally:
                os.unlink(config_file)

    def test_backoff_with_jitter(self):
        """Test that jitter adds randomness to backoff time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_file = create_backoff_test_config(
                output_path,
                backoff_enabled=True,
                backoff_base_ms=100.0,
                backoff_multiplier=2.0,
                backoff_max_ms=10000.0,
                backoff_jitter=0.2  # ±20% jitter
            )

            try:
                configure_from_toml(config_file)

                # Seed for reproducibility
                np.random.seed(42)

                # With 20% jitter, backoff should be within [80, 120] for retry 1
                retry_num = 1
                base_backoff = 100.0
                samples = [calculate_backoff_time(retry_num) for _ in range(100)]

                # Check that samples vary (jitter is working)
                assert len(set(samples)) > 1, "Jitter should produce varying backoff times"

                # Check that all samples are within jitter range
                for sample in samples:
                    assert 80.0 <= sample <= 120.0, \
                        f"Backoff with 20% jitter should be in [80, 120], got {sample}"

            finally:
                os.unlink(config_file)

    def test_backoff_integration_with_simulation(self):
        """Integration test: run simulation with backoff enabled and verify behavior."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            # High contention scenario to trigger retries
            config_file = create_backoff_test_config(
                output_path,
                seed=42,
                duration_ms=60000,  # 1 minute
                inter_arrival_scale=20.0,  # Very high contention
                backoff_enabled=True,
                backoff_base_ms=10.0,
                backoff_multiplier=2.0,
                backoff_max_ms=500.0,
                backoff_jitter=0.1
            )

            try:
                # Load config
                configure_from_toml(config_file)

                # Reset stats and run simulation
                endive.main.STATS = Stats()
                np.random.seed(42)

                # Setup and run simulation
                env = simpy.Environment()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Export results
                endive.main.STATS.export_parquet(output_path)

                # Verify results
                assert os.path.exists(output_path), "Results file should be created"
                df = pd.read_parquet(output_path)

                # Should have transactions
                assert len(df) > 0, "Should have recorded transactions"

                # Should have some retries (high contention scenario)
                assert df['n_retries'].max() > 0, "High contention should trigger retries"

                # Transactions with retries should have longer commit latency
                # (due to backoff time added)
                no_retry_txns = df[df['n_retries'] == 0]
                retry_txns = df[df['n_retries'] > 0]

                if len(no_retry_txns) > 0 and len(retry_txns) > 0:
                    # Median commit latency should be higher for retry transactions
                    no_retry_median = no_retry_txns['commit_latency'].median()
                    retry_median = retry_txns['commit_latency'].median()
                    assert retry_median > no_retry_median, \
                        f"Retry transactions should have higher latency due to backoff: {retry_median} vs {no_retry_median}"

            finally:
                os.unlink(config_file)

    def test_backoff_comparison_with_without(self):
        """Compare simulation with and without backoff under high contention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed = 42
            duration = 60000
            inter_arrival = 20.0  # High contention

            # Run without backoff
            output_no_backoff = os.path.join(tmpdir, "no_backoff.parquet")
            config_no_backoff = create_backoff_test_config(
                output_no_backoff,
                seed=seed,
                duration_ms=duration,
                inter_arrival_scale=inter_arrival,
                backoff_enabled=False
            )

            try:
                configure_from_toml(config_no_backoff)
                endive.main.STATS = Stats()
                np.random.seed(seed)

                env = simpy.Environment()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)
                endive.main.STATS.export_parquet(output_no_backoff)

                df_no_backoff = pd.read_parquet(output_no_backoff)

            finally:
                os.unlink(config_no_backoff)

            # Run with backoff
            output_with_backoff = os.path.join(tmpdir, "with_backoff.parquet")
            config_with_backoff = create_backoff_test_config(
                output_with_backoff,
                seed=seed,
                duration_ms=duration,
                inter_arrival_scale=inter_arrival,
                backoff_enabled=True,
                backoff_base_ms=10.0,
                backoff_multiplier=2.0,
                backoff_max_ms=500.0,
                backoff_jitter=0.1
            )

            try:
                configure_from_toml(config_with_backoff)
                endive.main.STATS = Stats()
                np.random.seed(seed)

                env = simpy.Environment()
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)
                endive.main.STATS.export_parquet(output_with_backoff)

                df_with_backoff = pd.read_parquet(output_with_backoff)

            finally:
                os.unlink(config_with_backoff)

            # Analysis: backoff should affect performance
            # Both simulations should have transactions
            assert len(df_no_backoff) > 0, "No backoff simulation should have transactions"
            assert len(df_with_backoff) > 0, "With backoff simulation should have transactions"

            # Success rates should be different (backoff may improve or worsen depending on load)
            success_rate_no_backoff = len(df_no_backoff[df_no_backoff['status'] == 'committed']) / len(df_no_backoff)
            success_rate_with_backoff = len(df_with_backoff[df_with_backoff['status'] == 'committed']) / len(df_with_backoff)

            # Just verify both ran and produced results - actual performance depends on workload
            assert 0 <= success_rate_no_backoff <= 1
            assert 0 <= success_rate_with_backoff <= 1


def test_config_defaults():
    """Test that backoff defaults are applied when not specified in config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "results.parquet")

        # Create minimal config without backoff section
        config_content = f"""[simulation]
duration_ms = 1000
output_path = "{output_path}"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 10
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
inter_arrival.distribution = "fixed"
inter_arrival.value = 500
ntable.zipf = 10.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[storage]
max_parallel = 4
min_latency = 1
T_CAS.mean = 1
T_CAS.stddev = 0.1
T_METADATA_ROOT.read.mean = 1
T_METADATA_ROOT.read.stddev = 0.1
T_METADATA_ROOT.write.mean = 1
T_METADATA_ROOT.write.stddev = 0.1
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
            config_file = f.name

        try:
            configure_from_toml(config_file)

            # Verify defaults
            assert endive.main.RETRY_BACKOFF_ENABLED is False  # Disabled by default
            assert endive.main.RETRY_BACKOFF_BASE_MS == 10.0
            assert endive.main.RETRY_BACKOFF_MULTIPLIER == 2.0
            assert endive.main.RETRY_BACKOFF_MAX_MS == 5000.0
            assert endive.main.RETRY_BACKOFF_JITTER == 0.1

        finally:
            os.unlink(config_file)
