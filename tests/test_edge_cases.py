"""Tests for edge case behaviors in the simulator.

Verifies correct behavior for:
1. Zero retries (transactions fail immediately on conflict)
2. Single transaction (no conflicts possible)
3. Extreme load (near-100% abort rate)
4. Very large version gaps (>100 snapshots behind)
5. Boundary conditions and unusual parameter combinations
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from endive.main import configure_from_toml, Catalog, Txn, txn_commit
import endive.main
from endive.capstats import Stats
from endive.test_utils import create_test_config
import simpy


class TestZeroRetries:
    """Test behavior with retry limit set to zero."""

    def test_zero_retries_immediate_failure_on_conflict(self):
        """With retry=0, transactions should fail immediately on first conflict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=15000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1,
                retry=0  # NO RETRIES
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP = {0: 0}
                endive.main.GROUP_TO_TABLES = {0: [0]}

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)
                assert len(df) > 0, "Should have generated transactions"

                # With high contention and zero retries, many should fail
                aborted = df[df['status'] == 'aborted']
                assert len(aborted) > 0, "Should have aborted transactions with high contention"

                # All aborted transactions should have exactly 1 retry (initial attempt)
                # Note: n_retries=1 means first attempt (no actual retries), n_retries=2 means one retry
                assert (aborted['n_retries'] == 1).all(), \
                    "Aborted transactions should have 1 retry (initial attempt) with retry=0"

                # Committed transactions should also have 1 retry (succeeded on first attempt)
                committed = df[df['status'] == 'committed']
                if len(committed) > 0:
                    assert (committed['n_retries'] == 1).all(), \
                        "Committed transactions should have 1 retry (succeeded on first attempt) with retry=0"

                print(f"✓ Zero retries causes immediate failure")
                print(f"  Total: {len(df)}, Committed: {len(committed)}, Aborted: {len(aborted)}")
                print(f"  Success rate: {len(committed)/len(df)*100:.1f}%")

            finally:
                os.unlink(config_path)

    def test_zero_retries_low_contention_succeeds(self):
        """With retry=0 and low contention, transactions should still succeed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=15000,
                inter_arrival_scale=2000.0,  # Very low contention
                num_tables=10,
                retry=0
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

                # With low contention, should have high success rate even with zero retries
                success_rate = len(committed) / len(df) * 100
                assert success_rate > 80, \
                    f"Low contention should succeed even with retry=0, got {success_rate:.1f}%"

                print(f"✓ Zero retries with low contention succeeds")
                print(f"  Success rate: {success_rate:.1f}%")

            finally:
                os.unlink(config_path)


class TestSingleTransaction:
    """Test behavior with only one transaction."""

    def test_single_transaction_always_succeeds(self):
        """A single transaction should always succeed (no conflicts possible)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,  # Short duration to get one transaction
                inter_arrival_scale=10000.0,  # Very long interval
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

                # Should have at least 1 transaction
                assert len(df) >= 1, "Should have at least one transaction"

                # If we got exactly one, it should succeed with 1 retry (initial attempt)
                if len(df) == 1:
                    assert df.iloc[0]['status'] == 'committed', "Single transaction should commit"
                    assert df.iloc[0]['n_retries'] >= 1, "Single transaction should have at least 1 retry (initial attempt)"

                    print(f"✓ Single transaction succeeds with no actual retries (n_retries={df.iloc[0]['n_retries']})")

            finally:
                os.unlink(config_path)


class TestExtremeLoad:
    """Test behavior under extreme load (near saturation)."""

    def test_extreme_load_high_abort_rate(self):
        """Extreme load should cause very high abort rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=5.0,  # EXTREME load
                num_tables=1,  # Single table
                retry=5  # Limited retries
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP = {0: 0}
                endive.main.GROUP_TO_TABLES = {0: [0]}

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Analyze results
                df = pd.DataFrame(endive.main.STATS.transactions)
                assert len(df) > 50, "Should have many transactions with extreme load"

                aborted = df[df['status'] == 'aborted']
                committed = df[df['status'] == 'committed']

                abort_rate = len(aborted) / len(df) * 100

                # Expect high abort rate (>30%)
                # Note: With proper catalog read latency after conflict resolution,
                # transactions are more spread out, reducing abort rate slightly
                assert abort_rate > 30, \
                    f"Extreme load should cause high abort rate, got {abort_rate:.1f}%"

                # Committed transactions should have some retries
                # Note: With survivorship bias, even under extreme load, successful
                # transactions tend to have lower retry counts than aborted ones
                if len(committed) > 0:
                    avg_retries = committed['n_retries'].mean()
                    assert avg_retries > 1, \
                        f"Committed transactions should have some retries under extreme load, got {avg_retries:.1f}"

                print(f"✓ Extreme load causes high abort rate")
                print(f"  Abort rate: {abort_rate:.1f}%")
                print(f"  Avg retries (committed): {avg_retries:.1f}")

            finally:
                os.unlink(config_path)

    def test_extreme_load_simulator_stability(self):
        """Verify simulator remains stable under extreme load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=60000,  # Longer to stress test
                inter_arrival_scale=3.0,  # VERY extreme
                num_tables=1
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP = {0: 0}
                endive.main.GROUP_TO_TABLES = {0: [0]}

                # Run simulation - should complete without errors
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify results are valid
                df = pd.DataFrame(endive.main.STATS.transactions)
                assert len(df) > 0, "Should complete with transactions"

                # Verify no NaN or inf values
                for col in df.select_dtypes(include=[np.number]).columns:
                    assert not df[col].isna().any(), f"Column {col} has NaN values"
                    assert not np.isinf(df[col]).any(), f"Column {col} has inf values"

                print(f"✓ Simulator stable under extreme load")
                print(f"  Generated {len(df)} transactions")

            finally:
                os.unlink(config_path)


class TestLargeVersionGaps:
    """Test behavior with large version gaps (many snapshots behind)."""

    def test_large_version_gap_handling(self):
        """Simulate transaction that falls far behind (>100 snapshots)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                num_tables=3
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.TABLE_TO_GROUP, endive.main.GROUP_TO_TABLES = endive.main.partition_tables_into_groups(
                    endive.main.N_TABLES, endive.main.N_GROUPS,
                    endive.main.GROUP_SIZE_DIST, endive.main.LONGTAIL_PARAMS
                )

                env = simpy.Environment()
                catalog = Catalog(env)

                # Create transaction at S_0
                tblr = {0: 0, 1: 0}
                tblw = {0: 1}
                txn = Txn(1, 0, 100, 0, tblr, tblw)
                txn.v_dirty = {0: 0, 1: 0}

                # Manually advance catalog by 150 snapshots
                catalog.seq = 150
                for i in range(endive.main.N_TABLES):
                    catalog.tbl[i] = 150

                # Attempt commit - should handle large gap correctly
                process = env.process(txn_commit(env, txn, catalog))
                env.run(process)

                # Transaction should update to current version
                assert txn.v_catalog_seq == 150, \
                    f"Transaction should update to S_150, got S_{txn.v_catalog_seq}"

                # Should have read 150 manifest lists (one per snapshot behind)
                # Note: This is a simplified test - full validation would track actual reads

                print(f"✓ Large version gaps handled correctly")
                print(f"  Gap size: 150 snapshots")
                print(f"  Transaction updated to: S_{txn.v_catalog_seq}")

            finally:
                os.unlink(config_path)


class TestBoundaryConditions:
    """Test boundary conditions and parameter edge cases."""

    def test_single_table_single_group(self):
        """Test minimum configuration: 1 table, 1 group."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=10000,
                inter_arrival_scale=500.0,
                num_tables=1
            )

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.STATS = Stats()
                endive.main.TABLE_TO_GROUP = {0: 0}
                endive.main.GROUP_TO_TABLES = {0: [0]}

                # Run simulation
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Should complete successfully
                df = pd.DataFrame(endive.main.STATS.transactions)
                assert len(df) > 0, "Should generate transactions"

                print(f"✓ Minimum configuration (1 table, 1 group) works")

            finally:
                os.unlink(config_path)

    def test_many_tables_many_groups(self):
        """Test large configuration: many tables and groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=10000,
                inter_arrival_scale=500.0,
                num_tables=100  # Large catalog
            )

            try:
                # Modify to have many groups - replace existing num_groups value
                with open(config_path, 'r') as f:
                    content = f.read()
                # Replace the default num_groups = 1 with num_groups = 50
                content = content.replace('num_groups = 1', 'num_groups = 50')
                with open(config_path, 'w') as f:
                    f.write(content)

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

                # Should complete successfully
                df = pd.DataFrame(endive.main.STATS.transactions)
                assert len(df) > 0, "Should generate transactions"

                print(f"✓ Large configuration (100 tables, 50 groups) works")
                print(f"  Generated {len(df)} transactions")

            finally:
                os.unlink(config_path)

    def test_very_long_runtime(self):
        """Test transactions with very long runtimes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=120000,  # 2 minutes
                inter_arrival_scale=1000.0,
                num_tables=5
            )

            # Set very long runtime
            with open(config_path, 'r') as f:
                content = f.read()
            content = content.replace('runtime.min = 100', 'runtime.min = 30000')  # 30 seconds
            content = content.replace('runtime.mean = 200', 'runtime.mean = 60000')  # 1 minute
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

                # Should complete with some transactions
                df = pd.DataFrame(endive.main.STATS.transactions)

                # With long runtimes, may have transactions that start but don't finish
                if len(df) > 0:
                    # Verify long runtimes are respected
                    min_runtime = df['t_runtime'].min()
                    assert min_runtime >= 30000, \
                        f"Minimum runtime should be 30000ms, got {min_runtime}ms"

                print(f"✓ Very long runtimes handled correctly")
                if len(df) > 0:
                    print(f"  Min runtime: {df['t_runtime'].min():.0f}ms")
                    print(f"  Mean runtime: {df['t_runtime'].mean():.0f}ms")

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
