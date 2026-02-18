"""Tests for the snapshot-based message-passing API.

These tests verify that:
1. CatalogSnapshot captures correct state
2. catalog.read() returns proper snapshots with latency
3. catalog.try_cas() returns CASResult with server-time snapshot
4. Transaction lifecycle uses snapshots correctly
"""

import pytest
import simpy
import tempfile
import os

from endive.snapshot import CatalogSnapshot, CASResult
from endive.transaction import Txn
from endive.main import (
    Catalog,
    configure_from_toml,
    rand_tbl_from_snapshot,
)


def create_test_config(output_path: str = "test_results.parquet", seed: int = 42) -> str:
    """Create a test configuration file."""
    config_content = f"""[simulation]
duration_ms = 10000
output_path = "{output_path}"
seed = {seed}

[catalog]
num_tables = 3

[transaction]
retry = 3
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5

inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0
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


@pytest.fixture
def minimal_config():
    """Create minimal config for testing."""
    config_file = create_test_config()
    configure_from_toml(config_file)
    os.unlink(config_file)
    return config_file


class TestCatalogSnapshot:
    """Tests for CatalogSnapshot immutability and correctness."""

    def test_snapshot_is_frozen(self, minimal_config):
        """CatalogSnapshot should be immutable."""
        snapshot = CatalogSnapshot(
            seq=1,
            tbl=(0, 1, 2),
            partition_seq=None,
            ml_offset=(0, 0, 0),
            partition_ml_offset=None,
            timestamp=1000,
        )

        # Cannot modify frozen dataclass
        with pytest.raises(AttributeError):
            snapshot.seq = 2

    def test_snapshot_captures_tuple_state(self, minimal_config):
        """Snapshot should contain tuples, not mutable lists."""
        snapshot = CatalogSnapshot(
            seq=5,
            tbl=(10, 20, 30),
            partition_seq=None,
            ml_offset=(100, 200, 300),
            partition_ml_offset=None,
            timestamp=5000,
        )

        # Verify tuple types
        assert isinstance(snapshot.tbl, tuple)
        assert isinstance(snapshot.ml_offset, tuple)
        assert snapshot.tbl == (10, 20, 30)


class TestCatalogRead:
    """Tests for catalog.read() message-passing API."""

    def test_read_returns_snapshot(self, minimal_config):
        """catalog.read() should return a CatalogSnapshot."""
        env = simpy.Environment()
        catalog = Catalog(env)

        # Modify catalog state
        catalog.seq = 5
        catalog.tbl[0] = 10
        catalog.tbl[1] = 20

        result = None
        def process():
            nonlocal result
            result = yield from catalog.read()

        env.process(process())
        env.run()

        assert isinstance(result, CatalogSnapshot)
        assert result.seq == 5
        assert result.tbl[0] == 10
        assert result.tbl[1] == 20
        assert result.timestamp == env.now

    def test_read_pays_latency(self, minimal_config):
        """catalog.read() should advance simulation time."""
        env = simpy.Environment()
        catalog = Catalog(env)

        start_time = env.now

        def process():
            yield from catalog.read()

        env.process(process())
        env.run()

        # Time should have advanced by CAS latency
        assert env.now > start_time

    def test_read_snapshot_is_independent(self, minimal_config):
        """Snapshot should not change when catalog changes."""
        env = simpy.Environment()
        catalog = Catalog(env)
        catalog.tbl[0] = 100

        snapshot = None
        def process():
            nonlocal snapshot
            snapshot = yield from catalog.read()
            # Modify catalog after snapshot taken
            catalog.tbl[0] = 999

        env.process(process())
        env.run()

        # Snapshot should have original value
        assert snapshot.tbl[0] == 100


class TestCatalogTryCas:
    """Tests for catalog.try_cas() message-passing API."""

    def test_try_cas_returns_result(self, minimal_config):
        """catalog.try_cas() should return a CASResult."""
        env = simpy.Environment()
        catalog = Catalog(env)

        # Create a simple transaction
        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=100,
            v_catalog_seq=0,
            v_tblr={0: 0},
            v_tblw={0: 1},
        )
        txn.v_dirty = {0: 0}
        txn.current_snapshot = catalog._create_snapshot()

        result = None
        def process():
            nonlocal result
            result = yield from catalog.try_cas(txn, txn.current_snapshot)

        env.process(process())
        env.run()

        assert isinstance(result, CASResult)
        assert result.success is True
        assert isinstance(result.snapshot, CatalogSnapshot)

    def test_try_cas_snapshot_has_server_time_state(self, minimal_config):
        """CASResult snapshot should capture state at server processing time."""
        env = simpy.Environment()
        catalog = Catalog(env)
        catalog.tbl[0] = 50

        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=100,
            v_catalog_seq=0,
            v_tblr={0: 0},
            v_tblw={0: 1},
        )
        txn.v_dirty = {0: 0}
        txn.current_snapshot = catalog._create_snapshot()

        result = None
        def process():
            nonlocal result
            result = yield from catalog.try_cas(txn, txn.current_snapshot)

        env.process(process())
        env.run()

        # Snapshot captures state at server time (before CAS applies changes)
        # For successful CAS, the snapshot is taken before tbl is updated
        # After CAS success, catalog.tbl[0] should be 1, but snapshot had 50
        assert result.snapshot.tbl[0] == 50


class TestRandTblFromSnapshot:
    """Tests for snapshot-based table selection."""

    def test_uses_snapshot_state(self, minimal_config):
        """rand_tbl_from_snapshot should use snapshot table versions."""
        import numpy as np
        np.random.seed(42)

        snapshot = CatalogSnapshot(
            seq=5,
            tbl=(100, 200, 300),  # High version numbers
            partition_seq=None,
            ml_offset=(0, 0, 0),
            partition_ml_offset=None,
            timestamp=1000,
        )

        tblr, tblw = rand_tbl_from_snapshot(snapshot)

        # Read versions should match snapshot
        for t, v in tblr.items():
            assert v == snapshot.tbl[t]

        # Write versions should be read + 1
        for t, v in tblw.items():
            assert v == snapshot.tbl[t] + 1


class TestTransactionSnapshotLifecycle:
    """Integration tests for transaction snapshot usage."""

    def test_txn_has_snapshot_fields(self, minimal_config):
        """Txn dataclass should have snapshot fields."""
        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=100,
            v_catalog_seq=0,
            v_tblr={0: 0},
            v_tblw={0: 1},
        )

        # Check snapshot fields exist
        assert hasattr(txn, 'start_snapshot')
        assert hasattr(txn, 'current_snapshot')
        assert txn.start_snapshot is None  # Default
        assert txn.current_snapshot is None  # Default

    def test_txn_stores_snapshots(self, minimal_config):
        """Transaction should store snapshots from catalog operations."""
        env = simpy.Environment()
        catalog = Catalog(env)

        snapshot = catalog._create_snapshot()

        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=100,
            v_catalog_seq=0,
            v_tblr={0: 0},
            v_tblw={0: 1},
        )
        txn.start_snapshot = snapshot
        txn.current_snapshot = snapshot

        assert txn.start_snapshot is snapshot
        assert txn.current_snapshot is snapshot
        assert txn.start_snapshot.seq == 0
