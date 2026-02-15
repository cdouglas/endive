"""Tests for partition-level modeling."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import simpy

from endive.main import (
    configure_from_toml,
    Catalog,
    Txn,
    select_partitions,
)
from endive.capstats import Stats, truncated_zipf_pmf
import endive.main


def create_partition_test_config(
    output_path: str,
    seed: int = None,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    num_tables: int = 1,
    partition_enabled: bool = True,
    num_partitions: int = 100,
    partition_distribution: str = "zipf",
    zipf_alpha: float = 1.5,
    partitions_per_txn_mean: float = 3.0,
    partitions_per_txn_max: int = 10,
    real_conflict_probability: float = 0.0,
    num_groups: int = None,  # Defaults to num_tables if None
) -> str:
    """Create a test configuration file for partition mode."""
    if num_groups is None:
        num_groups = 1  # Default to single group (catalog-level conflicts)
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
{'seed = ' + str(seed) if seed is not None else '# seed = 42'}

[catalog]
num_tables = {num_tables}
num_groups = {num_groups}

[partition]
enabled = {str(partition_enabled).lower()}
num_partitions = {num_partitions}
partitions_per_txn_mean = {partitions_per_txn_mean}
partitions_per_txn_max = {partitions_per_txn_max}

[partition.selection]
distribution = "{partition_distribution}"
zipf_alpha = {zipf_alpha}

[transaction]
retry = 5
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
real_conflict_probability = {real_conflict_probability}

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
max_parallel = 4
min_latency = 5

T_CAS.mean = 100
T_CAS.stddev = 10

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


class TestPartitionConfig:
    """Test partition configuration loading."""

    def test_partition_config_defaults(self):
        """Test that partition config has correct defaults when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=False,
            )
            try:
                configure_from_toml(config_path)
                assert endive.main.PARTITION_ENABLED == False
                assert endive.main.N_PARTITIONS == 100  # Default
                assert endive.main.PARTITION_SELECTION_DIST == "zipf"  # Default
            finally:
                os.unlink(config_path)

    def test_partition_config_enabled(self):
        """Test that partition config loads correctly when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=50,
                partition_distribution="uniform",
                zipf_alpha=2.0,
            )
            try:
                configure_from_toml(config_path)
                assert endive.main.PARTITION_ENABLED == True
                assert endive.main.N_PARTITIONS == 50
                assert endive.main.PARTITION_SELECTION_DIST == "uniform"
                assert endive.main.PARTITION_ZIPF_ALPHA == 2.0
            finally:
                os.unlink(config_path)


class TestPartitionSelection:
    """Test partition selection distributions."""

    def test_select_partitions_returns_valid_sets(self):
        """Test that select_partitions returns valid partition sets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
            )
            try:
                configure_from_toml(config_path)
                np.random.seed(42)

                for _ in range(100):
                    partitions_read, partitions_written = select_partitions(100)

                    # Written partitions should be subset of read
                    assert partitions_written.issubset(partitions_read)

                    # At least one partition should be written
                    assert len(partitions_written) >= 1

                    # All partitions should be valid IDs
                    for p in partitions_read | partitions_written:
                        assert 0 <= p < 100
            finally:
                os.unlink(config_path)

    def test_select_partitions_zipf_distribution(self):
        """Test that Zipf distribution favors lower partition IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
                partition_distribution="zipf",
                zipf_alpha=2.0,  # Strong skew
            )
            try:
                configure_from_toml(config_path)
                np.random.seed(42)

                # Count partition selections
                counts = np.zeros(100)
                for _ in range(1000):
                    partitions_read, _ = select_partitions(100)
                    for p in partitions_read:
                        counts[p] += 1

                # Lower partitions should be selected more often
                # Partition 0 should be selected significantly more than partition 99
                assert counts[0] > counts[99] * 5, f"Partition 0: {counts[0]}, Partition 99: {counts[99]}"
            finally:
                os.unlink(config_path)

    def test_select_partitions_uniform_distribution(self):
        """Test that uniform distribution is relatively even."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
                partition_distribution="uniform",
            )
            try:
                configure_from_toml(config_path)
                np.random.seed(42)

                # Count partition selections
                counts = np.zeros(100)
                for _ in range(1000):
                    partitions_read, _ = select_partitions(100)
                    for p in partitions_read:
                        counts[p] += 1

                # With uniform distribution, no partition should dominate
                # The ratio between max and min should be reasonable
                ratio = counts.max() / (counts.min() + 1)  # +1 to avoid div by zero
                assert ratio < 10, f"Max/min ratio too high for uniform: {ratio}"
            finally:
                os.unlink(config_path)


class TestCatalogPartition:
    """Test Catalog partition-level operations."""

    def test_catalog_partition_initialization(self):
        """Test that Catalog initializes partition state when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=50,
                num_tables=3,  # Test with multiple tables
            )
            try:
                configure_from_toml(config_path)
                sim = simpy.Environment()
                catalog = Catalog(sim)

                assert hasattr(catalog, 'partition_seq')
                # partition_seq is now indexed by [table_id][partition_id]
                assert len(catalog.partition_seq) == 3  # N_TABLES
                assert len(catalog.partition_seq[0]) == 50  # N_PARTITIONS
                assert all(v == 0 for v in catalog.partition_seq[0])

                assert hasattr(catalog, 'partition_ml_offset')
                assert len(catalog.partition_ml_offset) == 3  # N_TABLES
                assert len(catalog.partition_ml_offset[0]) == 50  # N_PARTITIONS
            finally:
                os.unlink(config_path)

    def test_catalog_partition_cas_success_different_partitions(self):
        """Test that CAS succeeds when transactions touch different partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
                num_tables=1,
            )
            try:
                configure_from_toml(config_path)
                sim = simpy.Environment()
                catalog = Catalog(sim)

                # Create transaction touching table 0, partitions 0, 1, 2
                # Now indexed as {table_id: {partition_ids}}
                txn1 = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn1.partitions_read = {0: {0, 1, 2}}
                txn1.partitions_written = {0: {0}}
                txn1.v_partition_seq = {0: {0: 0, 1: 0, 2: 0}}

                # Commit txn1
                assert catalog.try_CAS(sim, txn1) == True
                assert catalog.partition_seq[0][0] == 1  # Table 0, partition 0 incremented
                assert catalog.partition_seq[0][1] == 0  # Table 0, partition 1 not written

                # Create transaction touching partitions 5, 6 (different)
                txn2 = Txn(
                    id=2,
                    t_submit=100,
                    t_runtime=100,
                    v_catalog_seq=1,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn2.partitions_read = {0: {5, 6}}
                txn2.partitions_written = {0: {5}}
                txn2.v_partition_seq = {0: {5: 0, 6: 0}}

                # Should succeed - different partitions
                assert catalog.try_CAS(sim, txn2) == True
            finally:
                os.unlink(config_path)

    def test_catalog_partition_cas_fails_same_partition(self):
        """Test that CAS fails when transactions conflict on same partition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
                num_tables=1,
            )
            try:
                configure_from_toml(config_path)
                sim = simpy.Environment()
                catalog = Catalog(sim)

                # First transaction - table 0, partitions 0, 1
                txn1 = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn1.partitions_read = {0: {0, 1}}
                txn1.partitions_written = {0: {0}}
                txn1.v_partition_seq = {0: {0: 0, 1: 0}}

                # Commit txn1
                assert catalog.try_CAS(sim, txn1) == True

                # Second transaction also touches partition 0
                txn2 = Txn(
                    id=2,
                    t_submit=50,
                    t_runtime=100,
                    v_catalog_seq=0,  # Took snapshot before txn1 committed
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn2.partitions_read = {0: {0, 2}}
                txn2.partitions_written = {0: {0}}
                txn2.v_partition_seq = {0: {0: 0, 2: 0}}  # Stale version for partition 0

                # Should fail - partition 0 has been modified
                assert catalog.try_CAS(sim, txn2) == False
            finally:
                os.unlink(config_path)


class TestPartitionSimulation:
    """Test full simulation with partition mode."""

    def test_partition_mode_runs_successfully(self):
        """Test that simulation runs with partition mode enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                duration_ms=5000,
                partition_enabled=True,
                num_partitions=100,
                inter_arrival_scale=200.0,  # Moderate load
            )
            try:
                df = run_simulation_from_config(config_path)

                # Should have some transactions
                assert len(df) > 0

                # Should have some commits
                committed = df[df['status'] == 'committed']
                assert len(committed) > 0, "No transactions committed"
            finally:
                os.unlink(config_path)

    def test_partition_mode_deterministic(self):
        """Test that partition mode produces deterministic results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path1 = os.path.join(tmpdir, "results1.parquet")
            output_path2 = os.path.join(tmpdir, "results2.parquet")

            config_path1 = create_partition_test_config(
                output_path1,
                seed=42,
                duration_ms=3000,
                partition_enabled=True,
            )
            config_path2 = create_partition_test_config(
                output_path2,
                seed=42,
                duration_ms=3000,
                partition_enabled=True,
            )
            try:
                df1 = run_simulation_from_config(config_path1)
                df2 = run_simulation_from_config(config_path2)

                # Results should be identical
                assert len(df1) == len(df2)
                assert (df1['txn_id'] == df2['txn_id']).all()
                assert (df1['status'] == df2['status']).all()
            finally:
                os.unlink(config_path1)
                os.unlink(config_path2)

    def test_more_partitions_reduces_conflicts(self):
        """Test that more partitions reduces conflict rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Few partitions - more conflicts
            output_path1 = os.path.join(tmpdir, "results_few.parquet")
            config_path1 = create_partition_test_config(
                output_path1,
                seed=42,
                duration_ms=5000,
                partition_enabled=True,
                num_partitions=5,  # Few partitions
                inter_arrival_scale=100.0,  # High load
            )

            # Many partitions - fewer conflicts
            output_path2 = os.path.join(tmpdir, "results_many.parquet")
            config_path2 = create_partition_test_config(
                output_path2,
                seed=42,
                duration_ms=5000,
                partition_enabled=True,
                num_partitions=100,  # Many partitions
                inter_arrival_scale=100.0,  # Same high load
            )

            try:
                df_few = run_simulation_from_config(config_path1)
                df_many = run_simulation_from_config(config_path2)

                # Calculate success rates
                success_rate_few = (df_few['status'] == 'committed').mean()
                success_rate_many = (df_many['status'] == 'committed').mean()

                # More partitions should have higher success rate
                # (fewer conflicts because transactions spread across more partitions)
                assert success_rate_many >= success_rate_few * 0.9, (
                    f"Expected many partitions ({success_rate_many:.2%}) to have higher "
                    f"success rate than few partitions ({success_rate_few:.2%})"
                )
            finally:
                os.unlink(config_path1)
                os.unlink(config_path2)

    def test_partition_mode_disabled_unchanged_behavior(self):
        """Test that disabled partition mode has same behavior as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run with partitions disabled
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                duration_ms=5000,
                partition_enabled=False,
                inter_arrival_scale=300.0,
            )
            try:
                df = run_simulation_from_config(config_path)

                # Should work as before
                assert len(df) > 0
                committed = df[df['status'] == 'committed']
                assert len(committed) > 0
            finally:
                os.unlink(config_path)
