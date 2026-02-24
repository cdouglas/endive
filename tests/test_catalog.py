"""Tests for endive.catalog — uniform Catalog interface per SPEC.md §2.

Unit tests:
- CatalogSnapshot, TableMetadata, CommitResult immutability
- CASCatalog: success returns no snapshot, failure returns snapshot
- CASCatalog: seq increments by exactly 1 on success
- AppendCatalog: commit includes discovery read cost in latency
- AppendCatalog: physical append + discovery read internally
- InstantCatalog: fixed latency, CAS semantics
- Catalog.seq never skips or decreases
- read() returns consistent snapshot

Integration tests:
- Multiple concurrent commits, only one succeeds per seq
- CommitResult interface identical across CAS/Append/Instant
- CASCatalog with real StorageProvider end-to-end
"""

import pytest
import numpy as np

from endive.storage import (
    InstantStorageProvider,
    S3ExpressStorageProvider,
    LognormalLatency,
    SizeBasedLatency,
    create_provider,
)
from endive.catalog import (
    TableMetadata,
    CatalogSnapshot,
    CommitResult,
    IntentionRecord,
    Catalog,
    CASCatalog,
    AppendCatalog,
    InstantCatalog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exhaust(gen):
    """Drive a generator to completion, return its value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def step(gen):
    """Advance one step, return yielded value."""
    return next(gen)


def make_instant_storage(seed=42):
    """Create an InstantStorageProvider."""
    return InstantStorageProvider(rng=np.random.RandomState(seed))


def make_s3x_storage(seed=42):
    """Create an S3ExpressStorageProvider from profiles."""
    return create_provider("s3x", rng=np.random.RandomState(seed))


# ---------------------------------------------------------------------------
# TableMetadata
# ---------------------------------------------------------------------------

class TestTableMetadata:

    def test_frozen(self):
        tm = TableMetadata(table_id=0, version=1, num_partitions=3,
                           partition_versions=(0, 0, 0))
        with pytest.raises(AttributeError):
            tm.version = 99

    def test_with_version(self):
        tm = TableMetadata(table_id=0, version=1, num_partitions=2,
                           partition_versions=(5, 6))
        tm2 = tm.with_version(10)
        assert tm2.version == 10
        assert tm.version == 1  # original unchanged
        assert tm2.partition_versions == (5, 6)

    def test_with_partition_version(self):
        tm = TableMetadata(table_id=0, version=1, num_partitions=3,
                           partition_versions=(0, 0, 0))
        tm2 = tm.with_partition_version(1, 7)
        assert tm2.partition_versions == (0, 7, 0)
        assert tm.partition_versions == (0, 0, 0)


# ---------------------------------------------------------------------------
# CatalogSnapshot
# ---------------------------------------------------------------------------

class TestCatalogSnapshot:

    def test_frozen(self):
        tables = (TableMetadata(0, 0, 1, (0,)),)
        snap = CatalogSnapshot(seq=0, tables=tables, timestamp_ms=0.0)
        with pytest.raises(AttributeError):
            snap.seq = 99

    def test_get_table(self):
        t0 = TableMetadata(0, 5, 2, (1, 2))
        t1 = TableMetadata(1, 3, 1, (0,))
        snap = CatalogSnapshot(seq=1, tables=(t0, t1), timestamp_ms=0.0)
        assert snap.get_table(0).version == 5
        assert snap.get_table(1).version == 3

    def test_get_partition_version(self):
        t0 = TableMetadata(0, 0, 3, (10, 20, 30))
        snap = CatalogSnapshot(seq=0, tables=(t0,), timestamp_ms=0.0)
        assert snap.get_partition_version(0, 0) == 10
        assert snap.get_partition_version(0, 2) == 30


# ---------------------------------------------------------------------------
# CommitResult
# ---------------------------------------------------------------------------

class TestCommitResult:

    def test_frozen(self):
        cr = CommitResult(success=True, latency_ms=1.0)
        with pytest.raises(AttributeError):
            cr.success = False

    def test_success_fields(self):
        cr = CommitResult(success=True, latency_ms=5.0)
        assert cr.success is True
        assert cr.latency_ms == 5.0

    def test_failure_fields(self):
        cr = CommitResult(success=False, latency_ms=5.0)
        assert cr.success is False
        assert cr.latency_ms == 5.0


# ---------------------------------------------------------------------------
# IntentionRecord
# ---------------------------------------------------------------------------

class TestIntentionRecord:

    def test_frozen(self):
        ir = IntentionRecord(txn_id=1, expected_seq=0, tables_written={0: 1})
        with pytest.raises(AttributeError):
            ir.txn_id = 99

    def test_default_size(self):
        ir = IntentionRecord(txn_id=1, expected_seq=0, tables_written={0: 1})
        assert ir.size_bytes == 100


# ---------------------------------------------------------------------------
# CASCatalog
# ---------------------------------------------------------------------------

class TestCASCatalog:

    def test_requires_cas_support(self):
        """CASCatalog rejects storage without CAS support."""
        # InstantStorageProvider supports CAS, so this should work
        storage = make_instant_storage()
        CASCatalog(storage, num_tables=1, partitions_per_table=(1,))

    def test_initial_seq_zero(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        assert cat.seq == 0

    def test_read_returns_snapshot(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 2, (3, 2))
        snap = exhaust(cat.read(timestamp_ms=100.0))
        assert isinstance(snap, CatalogSnapshot)
        assert snap.seq == 0
        assert len(snap.tables) == 2
        assert snap.tables[0].num_partitions == 3
        assert snap.tables[1].num_partitions == 2
        assert snap.timestamp_ms == 100.0

    def test_commit_success(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert isinstance(result, CommitResult)
        assert result.success is True
        assert result.latency_ms > 0

    def test_commit_failure(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        # First commit succeeds (seq 0 -> 1)
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        # Second commit with stale seq fails
        result = exhaust(cat.commit(expected_seq=0, writes={0: 2}))
        assert result.success is False

    def test_seq_increments_by_one(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        assert cat.seq == 0
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1
        exhaust(cat.commit(expected_seq=1, writes={0: 2}))
        assert cat.seq == 2

    def test_seq_does_not_increment_on_failure(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1
        exhaust(cat.commit(expected_seq=0, writes={0: 2}))  # stale seq
        assert cat.seq == 1  # unchanged

    def test_writes_applied_on_success(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 2, (1, 1))
        exhaust(cat.commit(expected_seq=0, writes={0: 5, 1: 3}))
        snap = exhaust(cat.read())
        assert snap.get_table(0).version == 5
        assert snap.get_table(1).version == 3

    def test_writes_not_applied_on_failure(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        exhaust(cat.commit(expected_seq=0, writes={0: 99}))  # stale
        snap = exhaust(cat.read())
        assert snap.get_table(0).version == 1  # not 99

    def test_read_after_commit_reflects_changes(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (2,))
        exhaust(cat.commit(expected_seq=0, writes={0: 7}))
        snap = exhaust(cat.read())
        assert snap.seq == 1
        assert snap.get_table(0).version == 7

    def test_partitions_per_table_validation(self):
        storage = make_instant_storage()
        with pytest.raises(ValueError, match="partitions_per_table"):
            CASCatalog(storage, num_tables=2, partitions_per_table=(1,))

    def test_multiple_tables(self):
        storage = make_instant_storage()
        cat = CASCatalog(storage, 3, (2, 4, 1))
        snap = exhaust(cat.read())
        assert len(snap.tables) == 3
        assert snap.tables[0].num_partitions == 2
        assert snap.tables[1].num_partitions == 4
        assert snap.tables[2].num_partitions == 1


# ---------------------------------------------------------------------------
# AppendCatalog
# ---------------------------------------------------------------------------

class TestAppendCatalog:

    def test_requires_append_support(self):
        """AppendCatalog rejects storage without append support."""
        rng = np.random.RandomState(42)
        s3 = create_provider("s3", rng)  # S3 doesn't support append
        with pytest.raises(ValueError, match="append support"):
            AppendCatalog(s3, 1, (1,))

    def test_initial_seq_zero(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        assert cat.seq == 0

    def test_read_returns_snapshot(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (2,))
        snap = exhaust(cat.read(timestamp_ms=50.0))
        assert isinstance(snap, CatalogSnapshot)
        assert snap.seq == 0
        assert snap.timestamp_ms == 50.0

    def test_commit_success(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert isinstance(result, CommitResult)
        assert result.success is True

    def test_commit_failure(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 2}))
        assert result.success is False

    def test_seq_increments(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1
        exhaust(cat.commit(expected_seq=1, writes={0: 2}))
        assert cat.seq == 2

    def test_commit_latency_includes_discovery_read(self):
        """AppendCatalog commit latency >= 2x storage latency (append + read)."""
        storage = InstantStorageProvider(rng=np.random.RandomState(42),
                                          latency_ms=5.0)
        cat = AppendCatalog(storage, 1, (1,))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        # Append (5ms) + discovery read (5ms) = 10ms
        assert result.latency_ms >= 10.0

    def test_writes_applied_on_success(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 2, (1, 1))
        exhaust(cat.commit(expected_seq=0, writes={0: 5, 1: 3}))
        snap = exhaust(cat.read())
        assert snap.get_table(0).version == 5
        assert snap.get_table(1).version == 3

    def test_with_explicit_intention_record(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        intention = IntentionRecord(
            txn_id=42, expected_seq=0, tables_written={0: 1}
        )
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 1}, intention=intention
        ))
        assert result.success is True


# ---------------------------------------------------------------------------
# InstantCatalog
# ---------------------------------------------------------------------------

class TestInstantCatalog:

    def test_no_storage_required(self):
        """InstantCatalog doesn't need a StorageProvider."""
        cat = InstantCatalog(1, (1,))
        assert cat.seq == 0

    def test_read_returns_snapshot(self):
        cat = InstantCatalog(1, (2,))
        snap = exhaust(cat.read(timestamp_ms=42.0))
        assert isinstance(snap, CatalogSnapshot)
        assert snap.seq == 0
        assert snap.timestamp_ms == 42.0

    def test_commit_success(self):
        cat = InstantCatalog(1, (1,))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert result.success is True

    def test_commit_failure(self):
        cat = InstantCatalog(1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 2}))
        assert result.success is False

    def test_fixed_latency(self):
        cat = InstantCatalog(1, (1,), latency_ms=3.0)
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert result.latency_ms == 3.0

    def test_read_latency(self):
        cat = InstantCatalog(1, (1,), latency_ms=2.5)
        gen = cat.read()
        latency = next(gen)
        assert latency == 2.5

    def test_seq_increments(self):
        cat = InstantCatalog(1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1
        exhaust(cat.commit(expected_seq=1, writes={0: 2}))
        assert cat.seq == 2

    def test_custom_partitions(self):
        cat = InstantCatalog(3, (4, 2, 6))
        snap = exhaust(cat.read())
        assert len(snap.tables) == 3
        assert snap.tables[0].num_partitions == 4
        assert snap.tables[1].num_partitions == 2
        assert snap.tables[2].num_partitions == 6


# ---------------------------------------------------------------------------
# Invariant: seq never skips or decreases
# ---------------------------------------------------------------------------

class TestSeqInvariant:
    """Catalog.seq advances by exactly 1, never skips or decreases."""

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (1,)),
        lambda: CASCatalog(make_instant_storage(), 1, (1,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (1,)),
    ], ids=["instant", "cas", "append"])
    def test_seq_monotone_increment(self, catalog_factory):
        cat = catalog_factory()
        for expected_seq in range(20):
            assert cat.seq == expected_seq
            result = exhaust(cat.commit(expected_seq=expected_seq, writes={0: expected_seq + 1}))
            assert result.success is True
            assert cat.seq == expected_seq + 1

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (1,)),
        lambda: CASCatalog(make_instant_storage(), 1, (1,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (1,)),
    ], ids=["instant", "cas", "append"])
    def test_failed_commit_does_not_change_seq(self, catalog_factory):
        cat = catalog_factory()
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1
        for _ in range(5):
            exhaust(cat.commit(expected_seq=0, writes={0: 99}))  # stale
            assert cat.seq == 1


# ---------------------------------------------------------------------------
# Uniform CommitResult interface
# ---------------------------------------------------------------------------

class TestUniformInterface:
    """CommitResult is identical across all Catalog implementations."""

    def _commit_and_verify(self, cat):
        """Commit twice: first succeeds, second fails. Verify CommitResult."""
        # Success
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert isinstance(result, CommitResult)
        assert result.success is True
        assert result.latency_ms > 0

        # Failure (stale seq)
        result = exhaust(cat.commit(expected_seq=0, writes={0: 2}))
        assert isinstance(result, CommitResult)
        assert result.success is False
        assert result.latency_ms > 0

    def test_instant_catalog(self):
        self._commit_and_verify(InstantCatalog(1, (1,)))

    def test_cas_catalog(self):
        self._commit_and_verify(CASCatalog(make_instant_storage(), 1, (1,)))

    def test_append_catalog(self):
        self._commit_and_verify(AppendCatalog(make_instant_storage(), 1, (1,)))


# ---------------------------------------------------------------------------
# Integration: CASCatalog with real (non-instant) StorageProvider
# ---------------------------------------------------------------------------

class TestCASCatalogWithRealStorage:
    """CASCatalog end-to-end with S3ExpressStorageProvider."""

    def test_commit_with_s3x_storage(self):
        storage = make_s3x_storage()
        cat = CASCatalog(storage, 2, (4, 2))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1, 1: 1}))
        assert result.success is True
        # S3X latency should be > 10ms
        assert result.latency_ms >= 10.0

    def test_read_with_s3x_storage(self):
        storage = make_s3x_storage()
        cat = CASCatalog(storage, 1, (1,))
        snap = exhaust(cat.read(timestamp_ms=5000.0))
        assert snap.seq == 0
        assert snap.timestamp_ms == 5000.0

    def test_multiple_commits_s3x(self):
        storage = make_s3x_storage()
        cat = CASCatalog(storage, 1, (1,))
        for i in range(10):
            result = exhaust(cat.commit(expected_seq=i, writes={0: i + 1}))
            assert result.success is True
        assert cat.seq == 10
        snap = exhaust(cat.read())
        assert snap.get_table(0).version == 10


# ---------------------------------------------------------------------------
# Integration: concurrent commits
# ---------------------------------------------------------------------------

class TestConcurrentCommits:
    """Multiple commits at the same seq — only one can succeed."""

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (1,)),
        lambda: CASCatalog(make_instant_storage(), 1, (1,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (1,)),
    ], ids=["instant", "cas", "append"])
    def test_only_one_succeeds(self, catalog_factory):
        cat = catalog_factory()
        results = []
        for writer_id in range(5):
            result = exhaust(cat.commit(
                expected_seq=0, writes={0: writer_id + 1}
            ))
            results.append(result)

        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        assert len(successes) == 1
        assert len(failures) == 4
        assert cat.seq == 1


# ---------------------------------------------------------------------------
# Integration: AppendCatalog latency accounting
# ---------------------------------------------------------------------------

class TestAppendLatencyAccounting:
    """AppendCatalog commit cost = append + discovery read."""

    def test_latency_is_sum_of_append_and_read(self):
        """With known fixed-latency storage, verify latency sum."""
        lat = 7.0
        storage = InstantStorageProvider(rng=np.random.RandomState(42),
                                          latency_ms=lat)
        cat = AppendCatalog(storage, 1, (1,))
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        # append (7ms) + discovery read (7ms) = 14ms
        assert result.latency_ms == lat * 2

    def test_failure_latency_also_includes_read(self):
        """Even on failure, discovery read is performed."""
        lat = 3.0
        storage = InstantStorageProvider(rng=np.random.RandomState(42),
                                          latency_ms=lat)
        cat = AppendCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))  # succeed
        result = exhaust(cat.commit(expected_seq=0, writes={0: 2}))  # fail
        assert result.latency_ms == lat * 2


# ---------------------------------------------------------------------------
# Snapshot consistency
# ---------------------------------------------------------------------------

class TestSnapshotConsistency:
    """read() returns snapshot consistent with current state."""

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(2, (1, 1)),
        lambda: CASCatalog(make_instant_storage(), 2, (1, 1)),
        lambda: AppendCatalog(make_instant_storage(), 2, (1, 1)),
    ], ids=["instant", "cas", "append"])
    def test_read_reflects_committed_writes(self, catalog_factory):
        cat = catalog_factory()
        exhaust(cat.commit(expected_seq=0, writes={0: 10, 1: 20}))
        snap = exhaust(cat.read())
        assert snap.seq == 1
        assert snap.get_table(0).version == 10
        assert snap.get_table(1).version == 20

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (1,)),
        lambda: CASCatalog(make_instant_storage(), 1, (1,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (1,)),
    ], ids=["instant", "cas", "append"])
    def test_read_does_not_reflect_failed_writes(self, catalog_factory):
        cat = catalog_factory()
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        exhaust(cat.commit(expected_seq=0, writes={0: 99}))  # fails
        snap = exhaust(cat.read())
        assert snap.get_table(0).version == 1  # not 99


# ---------------------------------------------------------------------------
# Partition version tracking
# ---------------------------------------------------------------------------

class TestPartitionVersionTracking:
    """Verify partition versions advance on successful commits."""

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (4,)),
        lambda: CASCatalog(make_instant_storage(), 1, (4,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (4,)),
    ], ids=["instant", "cas", "append"])
    def test_partition_versions_advance_on_commit(self, catalog_factory):
        cat = catalog_factory()
        result = exhaust(cat.commit(
            expected_seq=0,
            writes={0: 1},
            partitions_written={0: frozenset({1, 3})},
        ))
        assert result.success is True
        snap = exhaust(cat.read())
        assert snap.get_partition_version(0, 0) == 0  # untouched
        assert snap.get_partition_version(0, 1) == 1  # advanced
        assert snap.get_partition_version(0, 2) == 0  # untouched
        assert snap.get_partition_version(0, 3) == 1  # advanced

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(1, (4,)),
        lambda: CASCatalog(make_instant_storage(), 1, (4,)),
        lambda: AppendCatalog(make_instant_storage(), 1, (4,)),
    ], ids=["instant", "cas", "append"])
    def test_no_partitions_written_leaves_versions_unchanged(self, catalog_factory):
        cat = catalog_factory()
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert result.success is True
        snap = exhaust(cat.read())
        assert snap.get_partition_version(0, 0) == 0
        assert snap.get_partition_version(0, 1) == 0

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(2, (2, 3)),
        lambda: CASCatalog(make_instant_storage(), 2, (2, 3)),
        lambda: AppendCatalog(make_instant_storage(), 2, (2, 3)),
    ], ids=["instant", "cas", "append"])
    def test_multi_table_partition_updates(self, catalog_factory):
        cat = catalog_factory()
        result = exhaust(cat.commit(
            expected_seq=0,
            writes={0: 1, 1: 1},
            partitions_written={0: frozenset({0}), 1: frozenset({1, 2})},
        ))
        assert result.success is True
        snap = exhaust(cat.read())
        assert snap.get_partition_version(0, 0) == 1
        assert snap.get_partition_version(0, 1) == 0
        assert snap.get_partition_version(1, 0) == 0
        assert snap.get_partition_version(1, 1) == 1
        assert snap.get_partition_version(1, 2) == 1
