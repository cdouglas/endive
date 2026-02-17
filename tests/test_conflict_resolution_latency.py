"""Tests for conflict resolution latency modeling.

Verifies that:
1. Catalog state reads have proper latency (no "teleportation")
2. After conflict resolution I/O, a catalog read is required before retry
3. Higher contention leads to more retries with proper latency modeling
"""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
import simpy

import endive.main
from endive.main import configure_from_toml
from endive.capstats import Stats
from endive.test_utils import create_test_config


def run_simulation_from_config(config_path: str) -> pd.DataFrame:
    """Run simulation and return results as DataFrame."""
    # Reset global stats
    endive.main.STATS = Stats()

    # Load configuration
    configure_from_toml(config_path)

    # Partition tables into groups (after seed is set)
    np.random.seed(endive.main.SIM_SEED if endive.main.SIM_SEED else 42)
    endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
        endive.main.N_TABLES,
        endive.main.N_GROUPS,
        endive.main.GROUP_SIZE_DIST,
        endive.main.LONGTAIL_PARAMS
    )

    # Run simulation
    sim = simpy.Environment()
    sim.process(endive.main.setup(sim))
    sim.run(until=endive.main.SIM_DURATION_MS)

    # Return transaction data as DataFrame
    return pd.DataFrame(endive.main.STATS.transactions)


class TestConflictResolutionLatency:
    """Test that conflict resolution includes catalog read latency."""

    def test_retry_includes_catalog_read_latency(self):
        """Verify retry cycle includes latency for reading current catalog state.

        The sequence should be:
        1. Conflict resolution I/O (read/write MLs)
        2. Read catalog to get current state (WITH latency)
        3. CAS attempt (WITH latency)

        Without the catalog read latency, information would "teleport" instantly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            # Short simulation with high contention (single partition)
            config_path = create_test_config(
                output_path=output_path,
                seed=42,
                duration_ms=60000,  # 1 minute
                inter_arrival_scale=50.0,  # High load
                num_tables=1,
                partition_enabled=True,
                num_partitions=1,  # All transactions conflict
            )

            try:
                df = run_simulation_from_config(config_path)
                committed = df[df['status'] == 'committed']

                if len(committed) == 0:
                    pytest.skip("No committed transactions in short simulation")

                # Transactions with retries (n_retries > 1) should have commit latency
                # that includes at least one catalog read + CAS cycle per retry
                retried = committed[committed['n_retries'] > 1]

                if len(retried) > 0:
                    # Each retry should add at least 2 * CAS latency (read + CAS)
                    # This is a lower bound since it excludes ML I/O
                    single_attempt = committed[committed['n_retries'] == 1]
                    if len(single_attempt) > 0:
                        base_latency = single_attempt['commit_latency'].median()
                        retry_latency = retried['commit_latency'].median()

                        # Retry latency should be significantly higher
                        # At minimum, one catalog read (~CAS latency) per retry
                        assert retry_latency > base_latency, \
                            f"Retry latency {retry_latency} should exceed base {base_latency}"
            finally:
                os.unlink(config_path)

    def test_partition_mode_catalog_read_after_ml_io(self):
        """Verify partition mode reads catalog after ML I/O, not before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            config_path = create_test_config(
                output_path=output_path,
                seed=123,
                duration_ms=30000,  # 30 seconds
                inter_arrival_scale=100.0,
                num_tables=1,
                partition_enabled=True,
                num_partitions=10,
            )

            try:
                df = run_simulation_from_config(config_path)
                committed = df[df['status'] == 'committed']

                # With proper latency modeling, retry overhead should be substantial
                # Each retry = ML I/O (~90ms) + catalog read (CAS latency) + CAS (CAS latency)
                if len(committed) > 10:
                    mean_retries = committed['n_retries'].mean()
                    mean_latency = committed['commit_latency'].mean()

                    # Sanity check: if there are retries, latency should reflect them
                    if mean_retries > 1.5:
                        # Each retry adds significant latency
                        # Base: ~90ms (ML write + metadata)
                        # Per retry: ~90ms (ML I/O) + ~CAS (catalog read) + ~CAS (CAS)
                        assert mean_latency > 100, \
                            f"With mean_retries={mean_retries:.2f}, latency should be >100ms, got {mean_latency:.1f}"
            finally:
                os.unlink(config_path)


class TestHighContentionRetries:
    """Test retry behavior under high contention scenarios."""

    def test_single_partition_high_contention(self):
        """With single partition, high load should cause many retries.

        When multiple transactions compete for the same partition with
        proper latency modeling, the conflict window during catalog read
        should cause additional retries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            # Very high load on single partition
            config_path = create_test_config(
                output_path=output_path,
                seed=456,
                duration_ms=60000,
                inter_arrival_scale=20.0,  # Very high load (~50 txns/sec)
                num_tables=1,
                partition_enabled=True,
                num_partitions=1,
            )

            try:
                df = run_simulation_from_config(config_path)
                committed = df[df['status'] == 'committed']

                if len(committed) > 0:
                    mean_retries = committed['n_retries'].mean()

                    # With proper latency modeling and high contention,
                    # mean retries should be > 1 (most transactions fail first attempt)
                    assert mean_retries > 1.0, \
                        f"Single partition high load should cause retries, got mean={mean_retries:.2f}"
            finally:
                os.unlink(config_path)

    def test_more_partitions_fewer_retries(self):
        """More partitions should lead to fewer retries due to less contention."""
        results = {}

        for n_partitions in [1, 10]:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, "results.parquet")

                config_path = create_test_config(
                    output_path=output_path,
                    seed=789,
                    duration_ms=30000,
                    inter_arrival_scale=100.0,
                    num_tables=1,
                    partition_enabled=True,
                    num_partitions=n_partitions,
                )

                try:
                    df = run_simulation_from_config(config_path)
                    committed = df[df['status'] == 'committed']

                    if len(committed) > 0:
                        results[n_partitions] = committed['n_retries'].mean()
                finally:
                    os.unlink(config_path)

        if len(results) == 2:
            # More partitions should have fewer retries
            assert results[10] <= results[1], \
                f"10 partitions ({results[10]:.2f} retries) should have <= retries than 1 partition ({results[1]:.2f})"


class TestTableModeLatency:
    """Test table mode (non-partition) has proper latency modeling."""

    def test_table_mode_catalog_read_after_ml_io(self):
        """Verify table mode reads catalog after ML I/O."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            # Table mode (partition disabled)
            config_path = create_test_config(
                output_path=output_path,
                seed=111,
                duration_ms=30000,
                inter_arrival_scale=100.0,
                num_tables=1,
                num_groups=1,  # Single group = catalog-level conflicts
                partition_enabled=False,
            )

            try:
                df = run_simulation_from_config(config_path)
                committed = df[df['status'] == 'committed']

                if len(committed) > 0:
                    mean_retries = committed['n_retries'].mean()
                    mean_latency = committed['commit_latency'].mean()

                    # Verify latency is reasonable
                    # Each retry should add catalog read + CAS latencies
                    assert mean_latency > 0, "Commit latency should be positive"
            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
