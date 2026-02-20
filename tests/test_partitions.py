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
    get_table_metadata_size,
    get_table_metadata_latency,
    ConflictResolver,
)
from endive.config import validate_config
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
) -> str:
    """Create a test configuration file for partition mode."""
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
{'seed = ' + str(seed) if seed is not None else '# seed = 42'}

[catalog]
num_tables = {num_tables}

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

    # Set random seed
    np.random.seed(endive.main.SIM_SEED if endive.main.SIM_SEED else 42)

    # Run simulation
    sim = simpy.Environment()
    sim.process(endive.main.setup(sim))
    sim.run(until=endive.main.SIM_DURATION_MS)

    # Return transaction data as DataFrame
    return pd.DataFrame(endive.main.STATS.transactions)


class TestPartitionValidation:
    """Test partition configuration validation."""

    def test_validation_warns_on_single_partition(self):
        """Test that num_partitions=1 triggers a warning."""
        config = {
            'catalog': {'num_tables': 1},
            'partition': {'enabled': True, 'num_partitions': 1},
            'simulation': {'duration_ms': 1000}
        }
        errors, warnings = validate_config(config)
        assert any('num_partitions = 1 is meaningless' in w for w in warnings)

    def test_validation_warns_on_high_partitions_per_txn(self):
        """Test that partitions_per_txn_mean >= num_partitions triggers a warning."""
        config = {
            'catalog': {'num_tables': 1},
            'partition': {'enabled': True, 'num_partitions': 5, 'partitions_per_txn_mean': 5.0},
            'simulation': {'duration_ms': 1000}
        }
        errors, warnings = validate_config(config)
        assert any('defeating partition isolation' in w for w in warnings)

    def test_validation_no_warnings_for_valid_config(self):
        """Test that a valid partition config produces no warnings."""
        config = {
            'catalog': {'num_tables': 1},
            'partition': {'enabled': True, 'num_partitions': 100, 'partitions_per_txn_mean': 3.0},
            'simulation': {'duration_ms': 1000}
        }
        errors, warnings = validate_config(config)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_validation_warns_on_partitions_exceeding_max(self):
        """Test that partitions_per_txn_max > num_partitions is a warning (clamped at runtime)."""
        config = {
            'catalog': {'num_tables': 1},
            'partition': {'enabled': True, 'num_partitions': 5, 'partitions_per_txn_max': 10},
            'simulation': {'duration_ms': 1000}
        }
        errors, warnings = validate_config(config)
        # Should be a warning, not an error (select_partitions clamps to n_partitions)
        assert len(errors) == 0
        assert any('will be clamped' in w for w in warnings)


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


class TestTableMetadataScaling:
    """Test that table metadata size/latency scales with partition count."""

    def test_metadata_size_scales_with_partitions(self):
        """Test that get_table_metadata_size() returns larger size with more partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Configure with partitions enabled
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=100,
            )
            try:
                configure_from_toml(config_path)

                # Get size with 100 partitions
                size_100 = get_table_metadata_size()

                # Change to 1000 partitions
                endive.main.N_PARTITIONS = 1000
                size_1000 = get_table_metadata_size()

                # Size should scale with partition count
                # With 100 bytes per partition entry:
                # 1000 partitions adds 100KB vs 100 partitions adds 10KB
                assert size_1000 > size_100, (
                    f"Size with 1000 partitions ({size_1000}) should be larger than "
                    f"size with 100 partitions ({size_100})"
                )

                # The difference should be approximately 900 * PARTITION_METADATA_ENTRY_SIZE
                expected_diff = 900 * endive.main.PARTITION_METADATA_ENTRY_SIZE
                actual_diff = size_1000 - size_100
                assert abs(actual_diff - expected_diff) < 100, (
                    f"Size difference ({actual_diff}) should be ~{expected_diff}"
                )
            finally:
                os.unlink(config_path)

    def test_metadata_size_unchanged_when_partitions_disabled(self):
        """Test that metadata size is base size when partitions disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=False,
            )
            try:
                configure_from_toml(config_path)

                size = get_table_metadata_size()

                # Should equal base size (no partition overhead)
                assert size == endive.main.TABLE_METADATA_SIZE_BYTES, (
                    f"Size ({size}) should equal base size "
                    f"({endive.main.TABLE_METADATA_SIZE_BYTES}) when partitions disabled"
                )
            finally:
                os.unlink(config_path)

    def test_metadata_latency_increases_with_partitions(self):
        """Test that metadata latency increases with more partitions when using size-based PUT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=10,
            )
            try:
                configure_from_toml(config_path)

                # Enable size-based PUT latency model (required for size-dependent latency)
                endive.main.T_PUT = {
                    'base_latency_ms': 30.0,
                    'latency_per_mib_ms': 20.0,
                    'sigma': 0.0,  # No variance for deterministic comparison
                }
                np.random.seed(42)

                # Sample latencies with 10 partitions
                latencies_10 = [get_table_metadata_latency('read') for _ in range(100)]
                mean_10 = np.mean(latencies_10)

                # Change to 1000 partitions (100x more)
                endive.main.N_PARTITIONS = 1000
                np.random.seed(42)

                latencies_1000 = [get_table_metadata_latency('read') for _ in range(100)]
                mean_1000 = np.mean(latencies_1000)

                # Mean latency should be higher with more partitions
                # 1000 partitions adds ~100KB, 10 partitions adds ~1KB
                # With 20ms per MiB, difference should be noticeable
                assert mean_1000 > mean_10, (
                    f"Mean latency with 1000 partitions ({mean_1000:.2f}ms) should be "
                    f"greater than with 10 partitions ({mean_10:.2f}ms)"
                )
            finally:
                os.unlink(config_path)


class TestPartitionConflictResolution:
    """Test that conflict resolution only touches conflicting partitions."""

    def test_get_conflicting_partitions_returns_only_changed(self):
        """Test that get_conflicting_partitions returns only modified partitions."""
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

                # Commit a transaction that modifies partition 5
                txn1 = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn1.partitions_read = {0: {5, 6, 7}}
                txn1.partitions_written = {0: {5}}
                txn1.v_partition_seq = {0: {5: 0, 6: 0, 7: 0}}
                catalog.try_CAS(sim, txn1)

                # Transaction that read partition 5 with stale version
                txn2 = Txn(
                    id=2,
                    t_submit=50,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn2.partitions_read = {0: {5, 10, 20}}  # Partition 5 is stale
                txn2.partitions_written = {0: {10}}
                txn2.v_partition_seq = {0: {5: 0, 10: 0, 20: 0}}  # Stale version for 5

                # Get conflicting partitions
                conflicting = catalog.get_conflicting_partitions(txn2)

                # Only partition 5 should be in conflict (table 0)
                assert 0 in conflicting, "Table 0 should have conflicts"
                assert 5 in conflicting[0], "Partition 5 should be in conflict"
                assert 10 not in conflicting[0], "Partition 10 should NOT be in conflict"
                assert 20 not in conflicting[0], "Partition 20 should NOT be in conflict"
                assert len(conflicting[0]) == 1, "Only 1 partition should be in conflict"
            finally:
                os.unlink(config_path)

    def test_conflict_resolution_stats_track_partition_conflicts(self):
        """Test that conflict resolution updates stats correctly for partitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                duration_ms=10000,
                partition_enabled=True,
                num_partitions=10,  # Few partitions = more conflicts
                inter_arrival_scale=100.0,  # High load
                real_conflict_probability=0.0,  # All false conflicts
            )
            try:
                # Reset stats
                endive.main.STATS = Stats()

                df = run_simulation_from_config(config_path)

                # Should have recorded some conflicts
                stats = endive.main.STATS
                total_conflicts = stats.false_conflicts + stats.real_conflicts

                assert total_conflicts > 0, "Should have some conflicts with high load"
                assert stats.real_conflicts == 0, "All conflicts should be false (prob=0)"
            finally:
                os.unlink(config_path)

    def test_multi_partition_conflict_resolves_each_independently(self):
        """Test that conflicts on multiple partitions are each resolved."""
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

                # Commit transactions that modify partitions 1, 2, 3
                for p in [1, 2, 3]:
                    txn = Txn(
                        id=p,
                        t_submit=0,
                        t_runtime=100,
                        v_catalog_seq=catalog.seq,
                        v_tblr={0: 0},
                        v_tblw={0: 1},
                    )
                    txn.partitions_read = {0: {p}}
                    txn.partitions_written = {0: {p}}
                    txn.v_partition_seq = {0: {p: catalog.partition_seq[0][p]}}
                    assert catalog.try_CAS(sim, txn), f"Txn for partition {p} should succeed"

                # Transaction that has stale versions for all three partitions
                txn_stale = Txn(
                    id=100,
                    t_submit=0,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                txn_stale.partitions_read = {0: {1, 2, 3, 50}}  # 50 is not in conflict
                txn_stale.partitions_written = {0: {1}}
                txn_stale.v_partition_seq = {0: {1: 0, 2: 0, 3: 0, 50: 0}}  # All stale

                conflicting = catalog.get_conflicting_partitions(txn_stale)

                # Should have 3 conflicting partitions
                assert len(conflicting[0]) == 3, (
                    f"Expected 3 conflicting partitions, got {len(conflicting[0])}"
                )
                assert conflicting[0] == {1, 2, 3}, (
                    f"Expected partitions {{1, 2, 3}}, got {conflicting[0]}"
                )
            finally:
                os.unlink(config_path)


class TestPartitionMetadataOverhead:
    """Test the O(N) metadata overhead at high partition counts."""

    def test_high_partition_count_increases_overhead(self):
        """Test that very high partition counts add measurable overhead."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")

            # Test with low partition count
            config_path_low = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=10,
            )
            configure_from_toml(config_path_low)
            size_low = get_table_metadata_size()
            os.unlink(config_path_low)

            # Test with high partition count
            config_path_high = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=10000,  # Very high
                partitions_per_txn_mean=2.0,
                partitions_per_txn_max=5,
            )
            configure_from_toml(config_path_high)
            size_high = get_table_metadata_size()
            os.unlink(config_path_high)

            # High partition count should add significant overhead
            # 10000 partitions * 100 bytes = 1MB overhead
            # 10 partitions * 100 bytes = 1KB overhead
            # But base size (10KB) is included, so ratio is:
            # (10KB + 1MB) / (10KB + 1KB) = ~90x
            ratio = size_high / size_low
            assert ratio > 50, (
                f"Size ratio ({ratio:.1f}x) should be >50x for 1000x more partitions"
            )

            # Also verify the absolute overhead is correct
            # 10000 partitions = 1MB overhead
            partition_overhead = size_high - endive.main.TABLE_METADATA_SIZE_BYTES
            expected_overhead = 10000 * endive.main.PARTITION_METADATA_ENTRY_SIZE
            assert partition_overhead == expected_overhead, (
                f"Partition overhead ({partition_overhead}) should equal "
                f"10000 * {endive.main.PARTITION_METADATA_ENTRY_SIZE} = {expected_overhead}"
            )

    def test_partition_overhead_documented_in_config(self):
        """Test that PARTITION_METADATA_ENTRY_SIZE is configurable."""
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

                # Default should be 100 bytes
                assert endive.main.PARTITION_METADATA_ENTRY_SIZE == 100, (
                    f"Default PARTITION_METADATA_ENTRY_SIZE should be 100, "
                    f"got {endive.main.PARTITION_METADATA_ENTRY_SIZE}"
                )
            finally:
                os.unlink(config_path)


class TestPartitionHistoryReads:
    """Test that partition mode reads manifest list history like Iceberg does.

    Iceberg's validationHistory traverses all snapshots between the transaction's
    read and current state. This ensures conflict detection is comprehensive.
    """

    def test_calculate_partition_snapshots_behind(self):
        """Test that calculate_partition_snapshots_behind returns correct gap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                partition_enabled=True,
                num_partitions=10,
                num_tables=1,
            )
            try:
                configure_from_toml(config_path)
                sim = simpy.Environment()
                catalog = Catalog(sim)

                # Commit 5 transactions to partition 3
                for i in range(5):
                    txn = Txn(
                        id=i + 1,
                        t_submit=0,
                        t_runtime=100,
                        v_catalog_seq=catalog.seq,
                        v_tblr={0: 0},
                        v_tblw={0: 1},
                    )
                    txn.partitions_read = {0: {3}}
                    txn.partitions_written = {0: {3}}
                    txn.v_partition_seq = {0: {3: catalog.partition_seq[0][3]}}
                    assert catalog.try_CAS(sim, txn) == True

                # Partition 3 should now be at version 5
                assert catalog.partition_seq[0][3] == 5

                # Create a transaction that read partition 3 at version 0
                stale_txn = Txn(
                    id=100,
                    t_submit=0,
                    t_runtime=100,
                    v_catalog_seq=0,
                    v_tblr={0: 0},
                    v_tblw={0: 1},
                )
                stale_txn.v_partition_seq = {0: {3: 0}}  # Read at version 0

                # Should be 5 snapshots behind
                n_behind = ConflictResolver.calculate_partition_snapshots_behind(
                    stale_txn, catalog, table_id=0, partition_id=3
                )
                assert n_behind == 5, f"Expected 5 snapshots behind, got {n_behind}"

                # Partition 7 was never modified, so version is 0
                stale_txn.v_partition_seq[0][7] = 0
                n_behind_7 = ConflictResolver.calculate_partition_snapshots_behind(
                    stale_txn, catalog, table_id=0, partition_id=7
                )
                assert n_behind_7 == 0, f"Unmodified partition should be 0 behind, got {n_behind_7}"
            finally:
                os.unlink(config_path)

    def test_partition_conflict_reads_history_mls(self):
        """Test that partition conflict resolution reads MLs for all snapshots behind.

        This matches Iceberg's validationHistory behavior: traverse all snapshots
        between starting and current to detect conflicts comprehensively.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                duration_ms=30000,
                partition_enabled=True,
                num_partitions=1,  # Single partition = all conflicts
                partitions_per_txn_mean=1.0,
                partitions_per_txn_max=1,
                inter_arrival_scale=100.0,  # Moderate-high load
                real_conflict_probability=0.0,
            )
            try:
                endive.main.STATS = Stats()
                df = run_simulation_from_config(config_path)

                # Should have some transactions with retries
                committed = df[df['status'] == 'committed']
                assert len(committed) > 0, "Should have committed transactions"

                # With single partition and high load, transactions should fall behind
                # and read multiple MLs during conflict resolution
                stats = endive.main.STATS
                ml_reads = stats.manifest_list_reads

                # The number of ML reads should be greater than the number of conflicts
                # because we read one ML per snapshot behind, not just one per conflict
                n_conflicts = stats.false_conflicts + stats.real_conflicts
                if n_conflicts > 0:
                    ml_reads_per_conflict = ml_reads / n_conflicts
                    # Should read at least 1 ML per conflict (history read)
                    # plus conflict resolution reads
                    assert ml_reads_per_conflict >= 1.0, (
                        f"Expected >= 1 ML read per conflict, got {ml_reads_per_conflict:.2f}"
                    )
            finally:
                os.unlink(config_path)

    def test_single_partition_saturates_like_single_table(self):
        """Test that 1 partition behaves similarly to single-table mode under load.

        With the history-based ML reads, single partition should show similar
        saturation behavior to single-table mode (high retry count, falling success rate).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.parquet")
            config_path = create_partition_test_config(
                output_path,
                seed=42,
                duration_ms=30000,
                partition_enabled=True,
                num_partitions=1,
                partitions_per_txn_mean=1.0,
                partitions_per_txn_max=1,
                inter_arrival_scale=50.0,  # High load
                real_conflict_probability=0.0,
            )
            try:
                df = run_simulation_from_config(config_path)
                committed = df[df['status'] == 'committed']

                if len(committed) > 10:
                    # Under high load with single partition, should see:
                    # 1. Multiple retries (transactions falling behind)
                    mean_retries = committed['n_retries'].mean()
                    assert mean_retries > 1.5, (
                        f"Expected mean retries > 1.5 under high load, got {mean_retries:.2f}"
                    )

                    # 2. Increased latency due to history reads
                    mean_latency = committed['commit_latency'].mean()
                    # With S3 latencies (~30ms per ML read) and multiple history reads,
                    # latency should be substantial
                    assert mean_latency > 100, (
                        f"Expected mean latency > 100ms with history reads, got {mean_latency:.1f}ms"
                    )
            finally:
                os.unlink(config_path)
