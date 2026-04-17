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
    TailAppendCatalog,
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
        snap = exhaust(cat.read())
        exhaust(cat.commit(expected_seq=0, writes={0: 1},
                           expected_log_offset=snap.log_offset))
        assert cat.seq == 1
        snap = exhaust(cat.read())
        exhaust(cat.commit(expected_seq=1, writes={0: 2},
                           expected_log_offset=snap.log_offset))
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
        # AppendCatalog increments table versions (not sets to absolute value)
        assert snap.get_table(0).version == 1
        assert snap.get_table(1).version == 1

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
        # Split-yield: first yield is half-RTT
        latency1 = next(gen)
        assert latency1 == 1.25
        latency2 = next(gen)
        assert latency2 == 1.25

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
# Half-RTT commit semantics
# ---------------------------------------------------------------------------

class TestHalfRTTCommit:
    """CAS version check and seq increment happen at half-RTT (server-side)."""

    def test_cas_two_yield_structure(self):
        """CASCatalog.commit() yields twice, each ~half the CAS latency."""
        storage = make_s3x_storage()
        cat = CASCatalog(storage, 1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1})
        first = step(gen)
        second = step(gen)
        assert first > 0
        assert second > 0
        total = first + second
        assert abs(first - total / 2.0) < 1e-10

    def test_instant_two_yield_structure(self):
        """InstantCatalog.commit() yields twice, each exactly half the latency."""
        cat = InstantCatalog(1, (1,), latency_ms=4.0)
        gen = cat.commit(expected_seq=0, writes={0: 1})
        first = step(gen)
        second = step(gen)
        assert first == 2.0
        assert second == 2.0

    def test_cas_seq_incremented_at_half_rtt(self):
        """After first yield (half-RTT), server has applied the write."""
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1})
        assert cat.seq == 0
        step(gen)           # request reaches server
        gen.send(None)      # resume — server applies write
        assert cat.seq == 1  # incremented at half-RTT

    def test_instant_seq_incremented_at_half_rtt(self):
        """InstantCatalog also increments seq at half-RTT."""
        cat = InstantCatalog(1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1})
        assert cat.seq == 0
        step(gen)           # request reaches server
        gen.send(None)      # resume — server applies write
        assert cat.seq == 1

    def test_cas_seq_unchanged_on_failure_at_half_rtt(self):
        """Failed CAS does not increment seq at half-RTT."""
        storage = make_instant_storage()
        cat = CASCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1

        gen = cat.commit(expected_seq=0, writes={0: 2})  # stale
        step(gen)           # request reaches server
        gen.send(None)      # resume — server rejects
        assert cat.seq == 1  # unchanged

    def test_instant_seq_unchanged_on_failure_at_half_rtt(self):
        """Failed InstantCatalog commit does not increment seq at half-RTT."""
        cat = InstantCatalog(1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.seq == 1

        gen = cat.commit(expected_seq=0, writes={0: 2})  # stale
        step(gen)
        gen.send(None)
        assert cat.seq == 1

    def test_cas_total_latency_preserved(self):
        """Total latency across both yields equals the sampled CAS latency."""
        storage = make_s3x_storage()
        cat = CASCatalog(storage, 1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1})
        first = step(gen)
        second = step(gen)
        result = exhaust(gen)
        assert abs(first + second - result.latency_ms) < 1e-10


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
            snap = exhaust(cat.read())
            assert snap.seq == expected_seq
            result = exhaust(cat.commit(
                expected_seq=expected_seq,
                writes={0: expected_seq + 1},
                expected_log_offset=snap.log_offset,
            ))
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
# Bug 1: Physical offset check
# ---------------------------------------------------------------------------

class TestAppendPhysicalCheck:
    """Position-append: expected_log_offset must match catalog's EOF."""

    def test_stale_offset_causes_failure(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (4,))
        # First commit at offset 0 succeeds
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            expected_log_offset=0,
        ))
        assert result.success is True
        # Second commit with stale offset 0 (should be 100) fails
        result = exhaust(cat.commit(
            expected_seq=1, writes={0: 2},
            partitions_written={0: frozenset({1})},
            expected_log_offset=0,  # stale
        ))
        assert result.success is False

    def test_correct_offset_succeeds(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (4,))
        snap = exhaust(cat.read())
        assert snap.log_offset == 0
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            expected_log_offset=snap.log_offset,
        ))
        assert result.success is True

    def test_snapshot_carries_log_offset(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        snap = exhaust(cat.read())
        assert snap.log_offset == 0
        exhaust(cat.commit(expected_seq=0, writes={0: 1},
                           expected_log_offset=0))
        snap2 = exhaust(cat.read())
        assert snap2.log_offset == 100  # default IntentionRecord.size_bytes

    def test_log_offset_advances_only_on_physical_success(self):
        """Physical failure leaves offset unchanged."""
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (1,))
        exhaust(cat.commit(expected_seq=0, writes={0: 1},
                           expected_log_offset=0))
        snap = exhaust(cat.read())
        assert snap.log_offset == 100
        # Physical failure: stale offset
        exhaust(cat.commit(expected_seq=1, writes={0: 2},
                           expected_log_offset=0))
        snap2 = exhaust(cat.read())
        assert snap2.log_offset == 100  # unchanged


# ---------------------------------------------------------------------------
# Bug 2: Success determined at server-eval time
# ---------------------------------------------------------------------------

class TestAppendServerEvalTiming:
    """Success is captured at half-RTT eval, immune to post-eval mutations."""

    def test_success_determined_before_discovery_read(self):
        """Drive the generator manually: mutate catalog state between
        append eval and discovery read. CommitResult must reflect the
        pre-mutation outcome."""
        storage = InstantStorageProvider(
            rng=np.random.RandomState(0), latency_ms=10.0)
        cat = AppendCatalog(storage, 1, (4,))
        gen = cat.commit(expected_seq=0, writes={0: 1},
                         partitions_written={0: frozenset({0})},
                         expected_log_offset=0)
        # Yield 1: append half-RTT (5ms) — generator pauses BEFORE eval
        y1 = next(gen)
        assert y1 == 5.0
        assert cat.seq == 0  # eval hasn't happened yet
        # Yield 2: generator runs server eval, then pauses at return half
        y2 = next(gen)
        assert cat.seq == 1  # eval happened between yield 1 and yield 2
        # Now simulate an intervening commit by another writer
        cat._tables[0].version = 999
        cat._seq = 42
        # Yield 3, 4: discovery read (split-yield)
        y3 = next(gen)
        y4 = next(gen)
        # Exhaust the generator to get the result
        try:
            next(gen)
            assert False, "Expected StopIteration"
        except StopIteration as e:
            result = e.value
        # A's commit must still report success even though catalog
        # state was mutated after A's server eval
        assert result.success is True


# ---------------------------------------------------------------------------
# Bug 3: Split-yield structure
# ---------------------------------------------------------------------------

class TestAppendSplitYield:
    """AppendCatalog uses split-yield matching CASCatalog convention."""

    def test_read_yields_twice(self):
        storage = InstantStorageProvider(
            rng=np.random.RandomState(42), latency_ms=6.0)
        cat = AppendCatalog(storage, 1, (1,))
        gen = cat.read()
        first = next(gen)
        second = next(gen)
        assert first == 3.0  # half-RTT
        assert second == 3.0

    def test_commit_yields_four_times(self):
        """2 yields for append + 2 for discovery read = 4 total."""
        storage = InstantStorageProvider(
            rng=np.random.RandomState(42), latency_ms=4.0)
        cat = AppendCatalog(storage, 1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1},
                         expected_log_offset=0)
        yields = []
        try:
            while True:
                yields.append(next(gen))
        except StopIteration:
            pass
        assert len(yields) == 4
        assert yields[0] == 2.0  # append half-RTT
        assert yields[1] == 2.0
        assert yields[2] == 2.0  # discovery half-RTT
        assert yields[3] == 2.0

    def test_seq_updated_at_append_half_rtt(self):
        """seq advances between the first two yields (append server eval).
        After next(gen) #1, generator paused AT yield 1 (before eval).
        After next(gen) #2, generator ran eval and paused AT yield 2."""
        storage = InstantStorageProvider(
            rng=np.random.RandomState(42), latency_ms=4.0)
        cat = AppendCatalog(storage, 1, (1,))
        gen = cat.commit(expected_seq=0, writes={0: 1},
                         expected_log_offset=0)
        assert cat.seq == 0
        next(gen)  # yield 1: half-RTT, eval NOT yet run
        assert cat.seq == 0
        next(gen)  # yield 2: eval ran between yield 1 and yield 2
        assert cat.seq == 1


# ---------------------------------------------------------------------------
# Bug 4: Per-partition preconditions
# ---------------------------------------------------------------------------

class TestAppendPartitionPreconditions:
    """Disjoint-partition writes commute; overlapping writes conflict."""

    def test_disjoint_partitions_both_succeed(self):
        """T1 writes {0,2}, T2 writes {1} — disjoint, both succeed."""
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (4,))

        # T1 writes partitions {0, 2}
        snap0 = exhaust(cat.read())
        result1 = exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0, 2})},
            expected_log_offset=snap0.log_offset,
            intention=IntentionRecord(
                txn_id=1, expected_seq=0, tables_written={0: 1},
                partitions_written={0: (0, 2)},
                expected_partition_versions={0: {0: 0, 2: 0}},
            ),
        ))
        assert result1.success is True

        # T2 writes partition {1} — disjoint with T1
        snap1 = exhaust(cat.read())
        result2 = exhaust(cat.commit(
            expected_seq=snap1.seq, writes={0: 2},
            partitions_written={0: frozenset({1})},
            expected_log_offset=snap1.log_offset,
            intention=IntentionRecord(
                txn_id=2, expected_seq=snap1.seq, tables_written={0: 2},
                partitions_written={0: (1,)},
                expected_partition_versions={0: {1: 0}},
            ),
        ))
        assert result2.success is True

    def test_overlapping_partitions_second_fails(self):
        """Two commits to the same partition: second fails."""
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (4,))

        snap0 = exhaust(cat.read())
        result1 = exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            expected_log_offset=snap0.log_offset,
            intention=IntentionRecord(
                txn_id=1, expected_seq=0, tables_written={0: 1},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},
            ),
        ))
        assert result1.success is True

        # T2 writes same partition {0} with STALE version expectation
        snap1 = exhaust(cat.read())
        result2 = exhaust(cat.commit(
            expected_seq=snap1.seq, writes={0: 2},
            partitions_written={0: frozenset({0})},
            expected_log_offset=snap1.log_offset,
            intention=IntentionRecord(
                txn_id=2, expected_seq=snap1.seq, tables_written={0: 2},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},  # stale! now at v1
            ),
        ))
        assert result2.success is False

    def test_cross_table_disjoint_succeeds(self):
        """Writes to different tables always commute."""
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 2, (2, 2))

        snap0 = exhaust(cat.read())
        exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            expected_log_offset=snap0.log_offset,
            intention=IntentionRecord(
                txn_id=1, expected_seq=0, tables_written={0: 1},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},
            ),
        ))
        snap1 = exhaust(cat.read())
        result = exhaust(cat.commit(
            expected_seq=snap1.seq, writes={1: 1},
            partitions_written={1: frozenset({0})},
            expected_log_offset=snap1.log_offset,
            intention=IntentionRecord(
                txn_id=2, expected_seq=snap1.seq, tables_written={1: 1},
                partitions_written={1: (0,)},
                expected_partition_versions={1: {0: 0}},
            ),
        ))
        assert result.success is True


# ---------------------------------------------------------------------------
# Table version increment
# ---------------------------------------------------------------------------

class TestAppendTableVersionIncrement:
    """AppendCatalog increments table versions (two disjoint commits
    from the same snapshot both advance the table version)."""

    def test_two_disjoint_commits_advance_version(self):
        storage = make_instant_storage()
        cat = AppendCatalog(storage, 1, (4,))

        snap0 = exhaust(cat.read())

        # T1 writes partition {0}
        exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            expected_log_offset=snap0.log_offset,
            intention=IntentionRecord(
                txn_id=1, expected_seq=0, tables_written={0: 1},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},
            ),
        ))

        snap1 = exhaust(cat.read())
        assert snap1.get_table(0).version == 1

        # T2 writes partition {1} (disjoint)
        exhaust(cat.commit(
            expected_seq=snap1.seq, writes={0: 2},
            partitions_written={0: frozenset({1})},
            expected_log_offset=snap1.log_offset,
            intention=IntentionRecord(
                txn_id=2, expected_seq=snap1.seq, tables_written={0: 2},
                partitions_written={0: (1,)},
                expected_partition_versions={0: {1: 0}},
            ),
        ))

        snap2 = exhaust(cat.read())
        # Table version should be 2 (two successful commits), not 1
        assert snap2.get_table(0).version == 2
        assert snap2.get_partition_version(0, 0) == 1
        assert snap2.get_partition_version(0, 1) == 1


# ---------------------------------------------------------------------------
# Snapshot consistency
# ---------------------------------------------------------------------------

class TestSnapshotConsistency:
    """read() returns snapshot consistent with current state."""

    @pytest.mark.parametrize("catalog_factory", [
        lambda: InstantCatalog(2, (1, 1)),
        lambda: CASCatalog(make_instant_storage(), 2, (1, 1)),
    ], ids=["instant", "cas"])
    def test_read_reflects_committed_writes(self, catalog_factory):
        cat = catalog_factory()
        exhaust(cat.commit(expected_seq=0, writes={0: 10, 1: 20}))
        snap = exhaust(cat.read())
        assert snap.seq == 1
        assert snap.get_table(0).version == 10
        assert snap.get_table(1).version == 20

    def test_read_reflects_committed_writes_append(self):
        """AppendCatalog increments table versions, doesn't set absolutes."""
        cat = AppendCatalog(make_instant_storage(), 2, (1, 1))
        exhaust(cat.commit(expected_seq=0, writes={0: 10, 1: 20}))
        snap = exhaust(cat.read())
        assert snap.seq == 1
        assert snap.get_table(0).version == 1
        assert snap.get_table(1).version == 1

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


# ---------------------------------------------------------------------------
# Inlined metadata: effective latency and catalog size growth
# ---------------------------------------------------------------------------

class TestInlinedMetadataLatency:
    """Verify _effective_latency, initial size, and growth for InstantCatalog."""

    def test_non_inlined_returns_base_latency(self):
        """metadata_inlined=False → latency is always base, regardless of latency_per_kib_ms."""
        cat = InstantCatalog(
            1, (10,), latency_ms=5.0,
            metadata_inlined=False, latency_per_kib_ms=1.0,
        )
        gen = cat.read()
        assert next(gen) == 2.5  # half-RTT

    def test_inlined_zero_rate_returns_base_latency(self):
        """metadata_inlined=True but latency_per_kib_ms=0 → base latency only."""
        cat = InstantCatalog(
            1, (10,), latency_ms=5.0,
            metadata_inlined=True, latency_per_kib_ms=0.0,
        )
        gen = cat.read()
        assert next(gen) == 2.5  # half-RTT

    def test_initial_size_from_partitions(self):
        """Initial catalog_size_bytes = total_partitions * initial_partition_size_bytes."""
        cat = InstantCatalog(
            2, (10, 5), latency_ms=1.0,
            metadata_inlined=True, initial_partition_size_bytes=1024,
        )
        # 15 partitions * 1024 = 15360
        assert cat.catalog_size_bytes == 15360

    def test_initial_size_default(self):
        """Default initial_partition_size_bytes=2048."""
        cat = InstantCatalog(
            1, (4,), latency_ms=1.0, metadata_inlined=True,
        )
        assert cat.catalog_size_bytes == 4 * 2048

    def test_effective_latency_formula(self):
        """Verify latency = base + (size_bytes / 1024) * latency_per_kib_ms."""
        cat = InstantCatalog(
            1, (1,), latency_ms=2.0,
            metadata_inlined=True,
            initial_partition_size_bytes=1024,  # 1 KiB
            latency_per_kib_ms=3.0,
        )
        # Expected effective latency: 2.0 + (1024/1024) * 3.0 = 5.0. First yield = half.
        gen = cat.read()
        assert next(gen) == pytest.approx(2.5)

    def test_effective_latency_larger_catalog(self):
        """Verify with multi-table, multi-partition initial size."""
        cat = InstantCatalog(
            2, (10, 10), latency_ms=1.0,
            metadata_inlined=True,
            initial_partition_size_bytes=2048,  # 20 partitions * 2 KiB = 40 KiB
            latency_per_kib_ms=0.5,
        )
        # Expected effective latency: 1.0 + 40.0 * 0.5 = 21.0. First yield = half.
        gen = cat.read()
        assert next(gen) == pytest.approx(10.5)

    def test_commit_latency_matches_effective(self):
        """commit() uses _effective_latency for its total latency."""
        cat = InstantCatalog(
            1, (1,), latency_ms=2.0,
            metadata_inlined=True,
            initial_partition_size_bytes=2048,
            latency_per_kib_ms=1.0,
        )
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        # Expected: 2.0 + (2048/1024) * 1.0 = 4.0
        assert result.latency_ms == pytest.approx(4.0)


class TestCatalogSizeGrowth:
    """Verify catalog_size_bytes grows by commit_growth_bytes on success only."""

    def test_size_grows_on_successful_commit(self):
        cat = InstantCatalog(
            1, (1,), latency_ms=1.0,
            metadata_inlined=True,
            initial_partition_size_bytes=1024,
            commit_growth_bytes=200,
        )
        initial = cat.catalog_size_bytes
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.catalog_size_bytes == initial + 200

    def test_size_unchanged_on_failed_commit(self):
        cat = InstantCatalog(
            1, (1,), latency_ms=1.0,
            metadata_inlined=True,
            initial_partition_size_bytes=1024,
            commit_growth_bytes=200,
        )
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        size_after_first = cat.catalog_size_bytes
        # Stale seq → CAS failure
        exhaust(cat.commit(expected_seq=0, writes={0: 2}))
        assert cat.catalog_size_bytes == size_after_first

    def test_size_after_n_commits(self):
        """After N successful commits: initial + N * growth."""
        cat = InstantCatalog(
            1, (2,), latency_ms=1.0,
            metadata_inlined=True,
            initial_partition_size_bytes=512,
            commit_growth_bytes=100,
        )
        initial = cat.catalog_size_bytes  # 2 * 512 = 1024
        assert initial == 1024
        n = 10
        for i in range(n):
            exhaust(cat.commit(expected_seq=i, writes={0: i + 1}))
        assert cat.catalog_size_bytes == initial + n * 100

    def test_latency_increases_with_growth(self):
        """Each commit should increase the effective latency."""
        cat = InstantCatalog(
            1, (1,), latency_ms=0.0,
            metadata_inlined=True,
            initial_partition_size_bytes=0,
            commit_growth_bytes=1024,  # +1 KiB per commit
            latency_per_kib_ms=1.0,
        )
        latencies = []
        for i in range(5):
            gen = cat.read()
            latencies.append(next(gen))
            exhaust(gen)
            exhaust(cat.commit(expected_seq=i, writes={0: i + 1}))
        # Latency should be monotonically increasing
        for j in range(1, len(latencies)):
            assert latencies[j] > latencies[j - 1], (
                f"Latency did not increase: {latencies}"
            )

    def test_non_inlined_size_stays_zero(self):
        """metadata_inlined=False → size stays at 0 regardless of commits."""
        cat = InstantCatalog(
            1, (10,), latency_ms=1.0,
            metadata_inlined=False,
            commit_growth_bytes=500,
        )
        assert cat.catalog_size_bytes == 0
        exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        assert cat.catalog_size_bytes == 0


# ---------------------------------------------------------------------------
# TailAppendCatalog
# ---------------------------------------------------------------------------

class TestTailAppendCatalog:

    def test_requires_tail_append_support(self):
        """TailAppendCatalog rejects storage without tail_append support."""
        rng = np.random.RandomState(42)
        s3x = create_provider("s3x", rng)  # S3X: supports_tail_append = False
        with pytest.raises(ValueError, match="tail_append"):
            TailAppendCatalog(s3x, 1, (1,))

    def test_sync_single_commit_succeeds(self):
        """Sync policy: single writer commits successfully."""
        storage = make_instant_storage()
        cat = TailAppendCatalog(storage, 1, (4,), compaction_policy="sync")
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 1},
            partitions_written={0: frozenset({0})},
            intention=IntentionRecord(
                txn_id=1, expected_seq=0, tables_written={0: 1},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},
            ),
        ))
        assert result.success is True
        assert cat.seq == 1

    def test_sync_seq_increments(self):
        """Sequential commits via sync drain-and-CAS."""
        storage = make_instant_storage()
        cat = TailAppendCatalog(storage, 1, (2,), compaction_policy="sync")
        for i in range(5):
            result = exhaust(cat.commit(
                expected_seq=i, writes={0: i + 1},
                partitions_written={0: frozenset({0})},
                intention=IntentionRecord(
                    txn_id=i + 1, expected_seq=i, tables_written={0: i + 1},
                    partitions_written={0: (0,)},
                    expected_partition_versions={0: {0: i}},
                ),
            ))
            assert result.success is True
        assert cat.seq == 5

    def test_sync_disjoint_partitions_both_succeed(self):
        """Sync: T1 writes {0}, T2 writes {1}. T1 drains queue with both
        intentions queued. Both succeed because partitions are disjoint."""
        storage = make_instant_storage()
        cat = TailAppendCatalog(storage, 1, (4,), compaction_policy="sync")

        # Queue T1's intention directly (simulating concurrent append)
        cat._queue.append((IntentionRecord(
            txn_id=1, expected_seq=0, tables_written={0: 1},
            partitions_written={0: (0,)},
            expected_partition_versions={0: {0: 0}},
        ), 0))
        cat._queue_counter = 1

        # T2 commits — will drain queue and process both
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 2},
            partitions_written={0: frozenset({1})},
            intention=IntentionRecord(
                txn_id=2, expected_seq=0, tables_written={0: 2},
                partitions_written={0: (1,)},
                expected_partition_versions={0: {1: 0}},
            ),
        ))
        assert result.success is True
        assert cat.seq == 2  # Both intentions applied
        snap = exhaust(cat.read())
        assert snap.get_partition_version(0, 0) == 1  # T1's write
        assert snap.get_partition_version(0, 1) == 1  # T2's write

    def test_sync_overlapping_fails_second(self):
        """Sync: two intentions to same partition. First succeeds (queue order),
        second fails because partition version advanced."""
        storage = make_instant_storage()
        cat = TailAppendCatalog(storage, 1, (4,), compaction_policy="sync")

        # Queue T1 first
        cat._queue.append((IntentionRecord(
            txn_id=1, expected_seq=0, tables_written={0: 1},
            partitions_written={0: (0,)},
            expected_partition_versions={0: {0: 0}},
        ), 0))
        cat._queue_counter = 1

        # T2 writes same partition {0} — will be second in queue order
        result = exhaust(cat.commit(
            expected_seq=0, writes={0: 2},
            partitions_written={0: frozenset({0})},
            intention=IntentionRecord(
                txn_id=2, expected_seq=0, tables_written={0: 2},
                partitions_written={0: (0,)},
                expected_partition_versions={0: {0: 0}},
            ),
        ))
        # T2 fails because T1 advanced partition 0 to version 1
        assert result.success is False
        assert cat.seq == 1  # Only T1 applied

    def test_sync_commit_latency(self):
        """Sync latency = append + read + CAS."""
        lat = 5.0
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=lat)
        cat = TailAppendCatalog(storage, 1, (1,), compaction_policy="sync",
                                compaction_read_latency_ms=lat)
        result = exhaust(cat.commit(expected_seq=0, writes={0: 1}))
        # append (5ms) + read (5ms) + CAS (5ms) = 15ms
        assert result.latency_ms == lat * 3

    def test_batched_single_commit(self):
        """Batched policy: writer waits for compaction cycle."""
        import simpy
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        cat = TailAppendCatalog(storage, 1, (4,), compaction_policy="batched",
                                compact_interval_ms=50.0)
        env = simpy.Environment()
        cat.setup(env)

        results = []
        def writer(env):
            gen = cat.commit(
                expected_seq=0, writes={0: 1},
                partitions_written={0: frozenset({0})},
                timestamp_ms=env.now,
                intention=IntentionRecord(
                    txn_id=1, expected_seq=0, tables_written={0: 1},
                    partitions_written={0: (0,)},
                    expected_partition_versions={0: {0: 0}},
                ),
            )
            latency = next(gen)
            while True:
                yield env.timeout(latency)
                try:
                    latency = gen.send(None)
                except StopIteration as e:
                    results.append(e.value)
                    break

        env.process(writer(env))
        env.run(until=200)

        assert len(results) == 1
        assert results[0].success is True
        assert cat.seq == 1

    def test_batched_latency_includes_compaction_wait(self):
        """Batched: commit latency > append latency (includes wait for cycle)."""
        import simpy
        lat = 1.0
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=lat)
        cat = TailAppendCatalog(storage, 1, (1,), compaction_policy="batched",
                                compact_interval_ms=50.0)
        env = simpy.Environment()
        cat.setup(env)

        results = []
        def writer(env):
            gen = cat.commit(
                expected_seq=0, writes={0: 1}, timestamp_ms=env.now,
            )
            latency = next(gen)
            while True:
                yield env.timeout(latency)
                try:
                    latency = gen.send(None)
                except StopIteration as e:
                    results.append(e.value)
                    break

        env.process(writer(env))
        env.run(until=200)

        assert len(results) == 1
        # Latency should be much more than just append (1ms) — includes wait for cycle
        assert results[0].latency_ms > 40  # at least interval/2 on average
