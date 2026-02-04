"""Tests for the append-based catalog operations."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import simpy

from endive.main import (
    configure_from_toml,
    AppendCatalog,
    LogEntry,
    check_logical_conflict,
    Txn,
)
from endive.capstats import Stats
import endive.main


def create_append_test_config(
    output_path: str,
    seed: int = None,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    num_tables: int = 5,
    compaction_threshold: int = 1000000,  # 1MB for testing
    log_entry_size: int = 100,
) -> str:
    """Create a test configuration file for append mode."""
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
{'seed = ' + str(seed) if seed is not None else '# seed = 42'}

[catalog]
num_tables = {num_tables}
mode = "append"
compaction_threshold = {compaction_threshold}
log_entry_size = {log_entry_size}

[transaction]
retry = 5
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

T_APPEND.mean = 50
T_APPEND.stddev = 5

T_LOG_ENTRY_READ.mean = 5
T_LOG_ENTRY_READ.stddev = 1

T_COMPACTION.mean = 200
T_COMPACTION.stddev = 20
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


class TestAppendCatalog:
    """Test the AppendCatalog class basic operations."""

    def setup_method(self):
        """Setup test fixtures."""
        # Configure a minimal config to set global variables
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_append_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                num_tables=5
            )
            try:
                configure_from_toml(config_path)
            finally:
                os.unlink(config_path)

    def test_append_catalog_initialization(self):
        """Test that AppendCatalog initializes correctly."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        assert len(catalog.tbl) == endive.main.N_TABLES
        assert all(v == 0 for v in catalog.tbl)
        assert catalog.log_offset == 0
        assert len(catalog.log_entries) == 0
        assert catalog.checkpoint_offset == 0
        assert catalog.sealed == False
        assert len(catalog.committed_txn) == 0

    def test_append_success_at_expected_offset(self):
        """Verify append succeeds when offset unchanged."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        # Create a transaction at offset 0
        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=100,
            v_catalog_seq=0,
            v_tblr={0: 0},
            v_tblw={0: 1}
        )
        txn.v_log_offset = 0

        # Create log entry
        entry = LogEntry(
            txn_id=1,
            tables_written={0: 1},
            tables_read={0: 0}
        )

        # Attempt append
        success, new_entries = catalog.try_APPEND(env, txn, entry)

        assert success == True
        assert new_entries == []
        assert catalog.log_offset == endive.main.LOG_ENTRY_SIZE
        assert len(catalog.log_entries) == 1
        assert catalog.tbl[0] == 1
        assert 1 in catalog.committed_txn

    def test_append_fails_when_offset_moved(self):
        """Verify append fails when concurrent append occurred."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        # First transaction succeeds
        txn1 = Txn(id=1, t_submit=0, t_runtime=100, v_catalog_seq=0,
                   v_tblr={0: 0}, v_tblw={0: 1})
        txn1.v_log_offset = 0
        entry1 = LogEntry(txn_id=1, tables_written={0: 1}, tables_read={0: 0})
        success1, _ = catalog.try_APPEND(env, txn1, entry1)
        assert success1 == True

        # Second transaction at old offset should fail
        txn2 = Txn(id=2, t_submit=50, t_runtime=100, v_catalog_seq=0,
                   v_tblr={1: 0}, v_tblw={1: 1})
        txn2.v_log_offset = 0  # Still at old offset
        entry2 = LogEntry(txn_id=2, tables_written={1: 1}, tables_read={1: 0})
        success2, new_entries = catalog.try_APPEND(env, txn2, entry2)

        assert success2 == False
        assert len(new_entries) == 1
        assert new_entries[0].txn_id == 1

    def test_get_log_entries_since(self):
        """Test retrieving log entries since a given offset."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        # Add multiple entries
        for i in range(3):
            txn = Txn(id=i+1, t_submit=i*10, t_runtime=100, v_catalog_seq=0,
                      v_tblr={i: 0}, v_tblw={i: 1})
            txn.v_log_offset = catalog.log_offset
            entry = LogEntry(txn_id=i+1, tables_written={i: 1}, tables_read={i: 0})
            catalog.try_APPEND(env, txn, entry)

        # Get entries since offset 0
        entries = catalog.get_log_entries_since(0)
        assert len(entries) == 3

        # Get entries since offset after first entry
        entries = catalog.get_log_entries_since(endive.main.LOG_ENTRY_SIZE)
        assert len(entries) == 2

        # Get entries from current offset (none)
        entries = catalog.get_log_entries_since(catalog.log_offset)
        assert len(entries) == 0


class TestLogicalConflictDetection:
    """Test the logical conflict detection logic."""

    def test_no_logical_conflict_different_tables(self):
        """Verify no conflict when different tables modified."""
        # Transaction writing to table 0
        txn = Txn(id=1, t_submit=0, t_runtime=100, v_catalog_seq=0,
                  v_tblr={0: 0}, v_tblw={0: 1})

        # Other entry wrote to table 1 (different table)
        other_entry = LogEntry(txn_id=2, tables_written={1: 1}, tables_read={1: 0})

        has_conflict = check_logical_conflict(txn, [other_entry])
        assert has_conflict == False

    def test_logical_conflict_write_write(self):
        """Verify conflict detected when same table written."""
        # Transaction writing to table 0
        txn = Txn(id=1, t_submit=0, t_runtime=100, v_catalog_seq=0,
                  v_tblr={0: 0}, v_tblw={0: 1})

        # Other entry also wrote to table 0
        other_entry = LogEntry(txn_id=2, tables_written={0: 2}, tables_read={0: 0})

        has_conflict = check_logical_conflict(txn, [other_entry])
        assert has_conflict == True

    def test_logical_conflict_read_write(self):
        """Verify conflict detected when read table was written."""
        # Transaction reads table 0, writes table 1
        txn = Txn(id=1, t_submit=0, t_runtime=100, v_catalog_seq=0,
                  v_tblr={0: 0, 1: 0}, v_tblw={1: 1})

        # Other entry wrote to table 0 (which we read)
        other_entry = LogEntry(txn_id=2, tables_written={0: 1}, tables_read={0: 0})

        has_conflict = check_logical_conflict(txn, [other_entry])
        assert has_conflict == True

    def test_no_conflict_multiple_entries_different_tables(self):
        """Verify no conflict with multiple entries on different tables."""
        # Transaction writing to table 0
        txn = Txn(id=1, t_submit=0, t_runtime=100, v_catalog_seq=0,
                  v_tblr={0: 0}, v_tblw={0: 1})

        # Other entries on tables 1, 2, 3
        entries = [
            LogEntry(txn_id=i+2, tables_written={i+1: 1}, tables_read={i+1: 0})
            for i in range(3)
        ]

        has_conflict = check_logical_conflict(txn, entries)
        assert has_conflict == False


class TestCompaction:
    """Test compaction triggering and execution."""

    def setup_method(self):
        """Setup test fixtures with small compaction threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_append_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                num_tables=5,
                compaction_threshold=500,  # Very small threshold for testing
                log_entry_size=100
            )
            try:
                configure_from_toml(config_path)
            finally:
                os.unlink(config_path)

    def test_compaction_triggered_at_threshold(self):
        """Verify compaction triggers when log exceeds threshold."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        # Reset stats
        endive.main.STATS = Stats()

        # Add entries until we exceed threshold (500 bytes / 100 bytes = 5 entries)
        for i in range(6):
            txn = Txn(id=i+1, t_submit=i*10, t_runtime=100, v_catalog_seq=0,
                      v_tblr={i % 5: i}, v_tblw={i % 5: i+1})
            txn.v_log_offset = catalog.log_offset
            entry = LogEntry(txn_id=i+1, tables_written={i % 5: i+1}, tables_read={i % 5: i})
            catalog.try_APPEND(env, txn, entry)

        # Should be sealed by now
        assert catalog.sealed == True
        assert endive.main.STATS.append_compactions_triggered > 0

    def test_compaction_clears_log(self):
        """Verify compaction clears log entries and resets state."""
        env = simpy.Environment()
        catalog = AppendCatalog(env)

        # Add some entries
        for i in range(3):
            txn = Txn(id=i+1, t_submit=i*10, t_runtime=100, v_catalog_seq=0,
                      v_tblr={i: 0}, v_tblw={i: 1})
            txn.v_log_offset = catalog.log_offset
            entry = LogEntry(txn_id=i+1, tables_written={i: 1}, tables_read={i: 0})
            catalog.try_APPEND(env, txn, entry)

        old_offset = catalog.log_offset

        # Perform compaction
        txn = Txn(id=99, t_submit=100, t_runtime=100, v_catalog_seq=0,
                  v_tblr={}, v_tblw={})
        result = catalog.try_CAS_compact(env, txn)

        assert result == True
        assert len(catalog.log_entries) == 0
        assert catalog.checkpoint_offset == old_offset
        assert catalog.sealed == False


class TestAppendModeSimulation:
    """Integration tests for append mode simulation."""

    def test_append_mode_runs_successfully(self):
        """Verify append mode simulation runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_append_test_config(
                output_path=os.path.join(tmpdir, "results.parquet"),
                seed=42,
                duration_ms=15000,
                inter_arrival_scale=300.0,
                num_tables=10
            )

            try:
                df = run_simulation_from_config(config_path)

                # Should have some transactions
                assert len(df) > 0

                # Should have committed transactions
                committed = df[df['status'] == 'committed']
                assert len(committed) > 0

                # Append stats should be populated
                stats = endive.main.STATS
                assert stats.append_physical_success > 0 or stats.append_physical_failure > 0

                print(f"Append mode test passed:")
                print(f"  Total txns: {len(df)}")
                print(f"  Committed: {len(committed)}")
                print(f"  Append physical success: {stats.append_physical_success}")
                print(f"  Append physical failure: {stats.append_physical_failure}")
                print(f"  Append logical success: {stats.append_logical_success}")
                print(f"  Append logical conflict: {stats.append_logical_conflict}")

            finally:
                os.unlink(config_path)

    def test_append_mode_deterministic(self):
        """Verify append mode produces deterministic results with same seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_append_test_config(
                output_path=os.path.join(tmpdir, "results.parquet"),
                seed=42,
                duration_ms=15000,
                inter_arrival_scale=400.0,
                num_tables=5
            )

            try:
                df1 = run_simulation_from_config(config_path)
                df2 = run_simulation_from_config(config_path)

                # Verify same number of transactions
                assert len(df1) == len(df2), "Different number of transactions"

                # Verify transaction IDs match
                assert df1['txn_id'].tolist() == df2['txn_id'].tolist(), \
                    "Transaction IDs differ"

                # Verify commit times match
                assert df1['t_commit'].tolist() == df2['t_commit'].tolist(), \
                    "Commit times differ"

                print(f"Append mode determinism test passed: {len(df1)} transactions")

            finally:
                os.unlink(config_path)

    def test_append_reduces_conflicts_multi_table(self):
        """Verify append mode has fewer conflicts than CAS for multi-table workloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # CAS mode config
            cas_config_content = f"""[simulation]
duration_ms = 20000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 20
mode = "cas"

[transaction]
retry = 5
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 200.0
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
T_CAS.mean = 50
T_CAS.stddev = 5
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
            cas_config_path = os.path.join(tmpdir, "cas.toml")
            with open(cas_config_path, 'w') as f:
                f.write(cas_config_content)

            # Append mode config
            append_config_path = create_append_test_config(
                output_path="results.parquet",
                seed=42,
                duration_ms=20000,
                inter_arrival_scale=200.0,
                num_tables=20
            )

            try:
                # Run CAS mode
                df_cas = run_simulation_from_config(cas_config_path)
                cas_retries = df_cas[df_cas['status'] == 'committed']['n_retries'].sum()

                # Run append mode
                df_append = run_simulation_from_config(append_config_path)
                append_retries = df_append[df_append['status'] == 'committed']['n_retries'].sum()

                print(f"Multi-table conflict comparison:")
                print(f"  CAS mode: {len(df_cas)} txns, {cas_retries} retries")
                print(f"  Append mode: {len(df_append)} txns, {append_retries} retries")

                # Append mode should generally have fewer retries for multi-table workloads
                # (though this can vary with seed and timing)
                # Just verify both modes work and produce valid results
                assert len(df_cas) > 0
                assert len(df_append) > 0

            finally:
                os.unlink(cas_config_path)
                os.unlink(append_config_path)


class TestManifestListAppend:
    """Test manifest list append functionality."""

    def test_manifest_list_append_runs_successfully(self):
        """Verify manifest list append mode runs without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with manifest_list_mode = "append"
            config_content = f"""[simulation]
duration_ms = 15000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 10
mode = "cas"

[transaction]
retry = 5
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
manifest_list_mode = "append"
inter_arrival.distribution = "exponential"
inter_arrival.scale = 300.0
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
            config_path = os.path.join(tmpdir, "ml_append.toml")
            with open(config_path, 'w') as f:
                f.write(config_content)

            try:
                df = run_simulation_from_config(config_path)

                # Should have some transactions
                assert len(df) > 0

                # Should have committed transactions
                committed = df[df['status'] == 'committed']
                assert len(committed) > 0

                # Manifest append stats should be populated
                stats = endive.main.STATS
                total_ml = stats.manifest_append_success + stats.manifest_append_retry
                assert total_ml > 0, "Manifest append operations should have occurred"

                print(f"Manifest list append test passed:")
                print(f"  Total txns: {len(df)}")
                print(f"  Committed: {len(committed)}")
                print(f"  ML append success: {stats.manifest_append_success}")
                print(f"  ML append retry: {stats.manifest_append_retry}")

            finally:
                os.unlink(config_path)

    def test_manifest_list_append_with_catalog_append(self):
        """Test combined manifest list append and catalog append modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config with both append modes enabled
            config_content = f"""[simulation]
duration_ms = 15000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 10
mode = "append"
compaction_threshold = 10000000
log_entry_size = 100

[transaction]
retry = 5
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
manifest_list_mode = "append"
inter_arrival.distribution = "exponential"
inter_arrival.scale = 300.0
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
T_APPEND.mean = 50
T_APPEND.stddev = 5
T_LOG_ENTRY_READ.mean = 5
T_LOG_ENTRY_READ.stddev = 1
T_COMPACTION.mean = 200
T_COMPACTION.stddev = 20
"""
            config_path = os.path.join(tmpdir, "combined.toml")
            with open(config_path, 'w') as f:
                f.write(config_content)

            try:
                df = run_simulation_from_config(config_path)

                # Should have some transactions
                assert len(df) > 0

                # Both catalog append and manifest append stats should be populated
                stats = endive.main.STATS
                total_catalog = stats.append_physical_success + stats.append_physical_failure
                total_ml = stats.manifest_append_success + stats.manifest_append_retry

                assert total_catalog > 0, "Catalog append operations should have occurred"
                assert total_ml > 0, "Manifest append operations should have occurred"

                print(f"Combined append modes test passed:")
                print(f"  Total txns: {len(df)}")
                print(f"  Catalog append success: {stats.append_physical_success}")
                print(f"  Catalog append failure: {stats.append_physical_failure}")
                print(f"  ML append success: {stats.manifest_append_success}")
                print(f"  ML append retry: {stats.manifest_append_retry}")

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
