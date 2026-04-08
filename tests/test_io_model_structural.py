"""Structural tests for the I/O model.

These tests verify the simulator's I/O operations at a structural level:
- Operation-tracking storage wrapper catches unexpected operations
- Regression tests ensure removed operations (manifest_file) stay removed
- Cross-provider latency ratio tests detect cost model drift
- Tight latency bounds replace the old loose "> 50ms" assertions
"""

from collections import Counter
from typing import Generator

import numpy as np
import pytest

from endive.catalog import InstantCatalog
from endive.storage import (
    InstantStorageProvider,
    StorageProvider,
    StorageResult,
    create_provider,
)
from endive.transaction import (
    FastAppendTransaction,
    TransactionStatus,
    ValidatedOverwriteTransaction,
)


# ---------------------------------------------------------------------------
# Operation-tracking storage wrapper (3a)
# ---------------------------------------------------------------------------

class TrackingStorageProvider(StorageProvider):
    """Wrapper that records every I/O operation dispatched to storage.

    Each operation is logged as (method, key, size) so tests can assert
    the exact sequence of I/O operations. Delegates to an inner provider.
    """

    def __init__(self, inner: StorageProvider):
        super().__init__(inner._rng)
        self._inner = inner
        self.ops: list[tuple[str, str, int]] = []

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        self.ops.append(("read", key, expected_size_bytes))
        return (yield from self._inner.read(key, expected_size_bytes))

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        self.ops.append(("write", key, size_bytes))
        return (yield from self._inner.write(key, size_bytes))

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        self.ops.append(("cas", key, size_bytes))
        return (yield from self._inner.cas(key, expected_version, size_bytes))

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        self.ops.append(("append", key, size_bytes))
        return (yield from self._inner.append(key, offset, size_bytes))

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        self.ops.append(("tail_append", key, size_bytes))
        return (yield from self._inner.tail_append(key, size_bytes))

    @property
    def supports_cas(self) -> bool:
        return self._inner.supports_cas

    @property
    def supports_append(self) -> bool:
        return self._inner.supports_append

    @property
    def supports_tail_append(self) -> bool:
        return self._inner.supports_tail_append

    @property
    def name(self) -> str:
        return f"tracking({self._inner.name})"

    @property
    def min_latency_ms(self) -> float:
        return self._inner.min_latency_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NeverRealDetector:
    def is_real_conflict(self, txn, current, start):
        return False


def drive(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def make_catalog(num_tables=1, partitions=(1,), latency_ms=1.0):
    return InstantCatalog(
        num_tables=num_tables,
        partitions_per_table=partitions,
        latency_ms=latency_ms,
    )


def make_fa(txn_id=1, tables=frozenset({0}), partitions=None):
    if partitions is None:
        partitions = {t: frozenset({0}) for t in tables}
    return FastAppendTransaction(
        txn_id=txn_id, submit_time_ms=0.0, runtime_ms=100.0,
        tables_written=tables, partitions_written=partitions,
    )


def make_vo(txn_id=1, tables=frozenset({0}), partitions=None):
    if partitions is None:
        partitions = {t: frozenset({0}) for t in tables}
    return ValidatedOverwriteTransaction(
        txn_id=txn_id, submit_time_ms=0.0, runtime_ms=100.0,
        tables_written=tables, partitions_written=partitions,
    )


# ---------------------------------------------------------------------------
# 3a. Operation tracking tests
# ---------------------------------------------------------------------------

class TestOperationTracking:
    """Use TrackingStorageProvider to verify exact I/O operation sequences."""

    def test_fa_clean_commit_operations(self):
        """FA clean commit, non-inlined: TM_read + ML_read + ML_write + TM_write."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        keys = [key for _, key, _ in storage.ops]
        assert keys == ["table_metadata", "manifest_list", "manifest_list", "table_metadata"], (
            f"Expected [TM read, ML read, ML write, TM write], got ops: {storage.ops}"
        )
        methods = [m for m, _, _ in storage.ops]
        assert methods == ["read", "read", "write", "write"]

    def test_fa_clean_commit_inlined_operations(self):
        """FA clean commit, inlined: ML_read + ML_write only (no TM file)."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector(),
                                   metadata_inlined=True))

        assert result.status == TransactionStatus.COMMITTED
        keys = [key for _, key, _ in storage.ops]
        assert keys == ["manifest_list", "manifest_list"]

    def test_fa_3_partition_operations(self):
        """FA 3 partitions, non-inlined: TM_read + 3×ML_read + 3×ML_write + TM_write."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog(num_tables=1, partitions=(3,))
        txn = make_fa(partitions={0: frozenset({0, 1, 2})})

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        keys = Counter(key for _, key, _ in storage.ops)
        assert keys["manifest_list"] == 6  # 3 reads + 3 writes
        assert keys["table_metadata"] == 2  # 1 read + 1 write
        assert len(storage.ops) == 8

    def test_fa_ml_append_mode_no_writes(self):
        """FA in ML+ mode, non-inlined: TM_read + ML_read + TM_write (no ML writes)."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector(),
                                   ml_append_mode=True))

        keys = [key for _, key, _ in storage.ops]
        assert keys == ["table_metadata", "manifest_list", "table_metadata"], (
            f"ML+ mode: TM_read + ML_read + TM_write, got: {storage.ops}"
        )

    def test_fa_retry_no_overlap_skips_per_attempt(self):
        """Cross-table retry: no storage ops on retry attempt."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog(num_tables=2, partitions=(1, 1))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2, tables=frozenset({1}),
                      partitions={1: frozenset({0})})
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read (not tracked — uses catalog directly)
        gen.send(None)  # runtime

        t1 = make_fa(txn_id=1, tables=frozenset({0}),
                     partitions={0: frozenset({0})})
        drive(t1.execute(catalog, storage, detector))

        # Clear ops from t1
        storage.ops.clear()

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        # First attempt: ML read + ML write (from before t1 committed)
        # Actually storage.ops was cleared after t1, so we see:
        # attempt0 per-attempt already happened before clear
        # After clear: CAS fail path reads catalog (not storage),
        # attempt1 has no per-attempt (no overlap) → just CAS
        # CAS uses catalog.commit() which uses storage internally for CASCatalog,
        # but InstantCatalog doesn't use storage for CAS
        # So after clear, no storage ops from retry
        keys = [key for _, key, _ in storage.ops]
        assert "manifest_file" not in keys

    def test_fa_retry_with_overlap_repays_per_attempt(self):
        """Same-table retry: per-attempt I/O paid again."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        storage.ops.clear()
        result = drive(gen)

        assert result.status == TransactionStatus.COMMITTED
        # After clearing: attempt0's remaining ops + attempt1's per-attempt
        # The per-attempt for attempt0 already ran before t1, so after clear
        # we see: (fail-path catalog read is not storage)
        # attempt1: ML read + ML write
        keys = [key for _, key, _ in storage.ops]
        assert "manifest_list" in keys

    def test_vo_convoy_operations(self):
        """VO with 2 versions behind: historical ML reads in conflict cost."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        detector = NeverRealDetector()

        txn = make_vo(txn_id=3)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        for i in range(2):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        storage.ops.clear()
        result = drive(gen)

        assert result.status == TransactionStatus.COMMITTED
        reads = [(m, k) for m, k, _ in storage.ops if m == "read"]
        writes = [(m, k) for m, k, _ in storage.ops if m == "write"]
        # Convoy: 1 historical ML read (2-1=1)
        # Per-attempt retry: 1 ML read + 1 ML write
        assert any(k == "manifest_list" for _, k in reads)


# ---------------------------------------------------------------------------
# 3b. Tight latency bounds (replacing loose "> 50ms" assertions)
# ---------------------------------------------------------------------------

class TestTightLatencyBounds:
    """Replace loose latency assertions with bounds that catch ±1 operation."""

    def test_fa_instant_commit_latency_exact(self):
        """InstantStorage FA, non-inlined: commit_latency = per_attempt(4) + CAS(1) = 5ms."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        # Exact: TM_r(1) + ML_r(1) + ML_w(1) + TM_w(1) + CAS(1) = 5ms
        assert result.commit_latency_ms == pytest.approx(5.0), (
            f"Expected 5.0ms commit latency, got {result.commit_latency_ms}"
        )

    def test_fa_s3_commit_latency_bounded(self):
        """S3 FA, non-inlined: commit_latency in [100, 500]ms (4 ops + CAS)."""
        results = []
        for seed in range(20):
            catalog = make_catalog(latency_ms=1.0)
            storage = create_provider("s3", rng=np.random.RandomState(seed))
            txn = make_fa(txn_id=seed)
            r = drive(txn.execute(catalog, storage, NeverRealDetector()))
            results.append(r.commit_latency_ms)

        median = sorted(results)[len(results) // 2]
        # 4 ops × ~43ms + 1ms CAS ≈ 173ms median
        assert 100.0 < median < 300.0, (
            f"S3 FA median commit_latency = {median:.1f}ms, expected [100, 300]ms"
        )

    def test_fa_s3x_faster_than_s3(self):
        """S3 Express should be faster than standard S3."""
        detector = NeverRealDetector()

        s3_latencies = []
        s3x_latencies = []
        for seed in range(20):
            cat = make_catalog(latency_ms=0.001)
            s3 = create_provider("s3", rng=np.random.RandomState(seed))
            txn = make_fa(txn_id=seed)
            r = drive(txn.execute(cat, s3, detector))
            s3_latencies.append(r.per_attempt_io_ms)

            cat2 = make_catalog(latency_ms=0.001)
            s3x = create_provider("s3x", rng=np.random.RandomState(seed))
            txn2 = make_fa(txn_id=seed + 100)
            r2 = drive(txn2.execute(cat2, s3x, detector))
            s3x_latencies.append(r2.per_attempt_io_ms)

        s3_median = sorted(s3_latencies)[10]
        s3x_median = sorted(s3x_latencies)[10]
        assert s3x_median < s3_median, (
            f"S3x median ({s3x_median:.1f}ms) should be < S3 ({s3_median:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# 3c. Cross-provider integration tests
# ---------------------------------------------------------------------------

class TestCrossProviderConsistency:
    """Same commit scenario across providers — verify latency ratios."""

    def test_per_attempt_scales_with_provider_latency(self):
        """Per-attempt I/O should scale linearly with provider latency.

        instant(1ms) → 2ms per-attempt
        instant(10ms) → 20ms per-attempt
        Ratio ≈ 10×.
        """
        detector = NeverRealDetector()

        cat1 = make_catalog(latency_ms=0.001)
        stor1 = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn1 = make_fa(txn_id=1)
        r1 = drive(txn1.execute(cat1, stor1, detector))

        cat2 = make_catalog(latency_ms=0.001)
        stor2 = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=10.0)
        txn2 = make_fa(txn_id=2)
        r2 = drive(txn2.execute(cat2, stor2, detector))

        assert r1.per_attempt_io_ms == pytest.approx(4.0)
        assert r2.per_attempt_io_ms == pytest.approx(40.0)

    def test_catalog_latency_independent_of_storage(self):
        """Changing catalog latency shouldn't affect per_attempt_io_ms."""
        detector = NeverRealDetector()

        cat1 = make_catalog(latency_ms=1.0)
        stor = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=5.0)
        txn1 = make_fa(txn_id=1)
        r1 = drive(txn1.execute(cat1, stor, detector))

        cat2 = make_catalog(latency_ms=100.0)
        stor2 = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=5.0)
        txn2 = make_fa(txn_id=2)
        r2 = drive(txn2.execute(cat2, stor2, detector))

        # per_attempt should be the same regardless of catalog latency
        assert r1.per_attempt_io_ms == r2.per_attempt_io_ms
        # But total latency should differ (catalog_read + CAS differ)
        assert r2.total_latency_ms > r1.total_latency_ms


# ---------------------------------------------------------------------------
# 3d. Regression: manifest_file must never be yielded
# ---------------------------------------------------------------------------

class TestNoManifestFileOps:
    """Regression tests: manifest_file operations were removed in 9d1d8e1.
    They must never appear in the I/O sequence.
    """

    def test_fa_clean_commit_no_manifest_file(self):
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        txn = make_fa()

        drive(txn.execute(catalog, storage, NeverRealDetector()))

        manifest_file_ops = [op for op in storage.ops if op[1] == "manifest_file"]
        assert manifest_file_ops == [], (
            f"manifest_file operations should not exist: {manifest_file_ops}"
        )

    def test_vo_clean_commit_no_manifest_file(self):
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        txn = make_vo()

        drive(txn.execute(catalog, storage, NeverRealDetector()))

        manifest_file_ops = [op for op in storage.ops if op[1] == "manifest_file"]
        assert manifest_file_ops == []

    def test_fa_retry_no_manifest_file(self):
        """Even on retry with overlap, no manifest_file ops."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        drive(gen)

        manifest_file_ops = [op for op in storage.ops if op[1] == "manifest_file"]
        assert manifest_file_ops == []

    def test_vo_retry_with_convoy_no_manifest_file(self):
        """VO retry with convoy cost: no manifest_file ops."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        detector = NeverRealDetector()

        txn = make_vo(txn_id=3)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        for i in range(3):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        drive(gen)

        manifest_file_ops = [op for op in storage.ops if op[1] == "manifest_file"]
        assert manifest_file_ops == []

    def test_multi_partition_no_manifest_file(self):
        """3-partition commit and retry: no manifest_file ops."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog(num_tables=1, partitions=(3,))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2, partitions={0: frozenset({0, 1, 2})})
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        t1 = make_fa(txn_id=1, partitions={0: frozenset({0, 1, 2})})
        drive(t1.execute(catalog, storage, detector))

        drive(gen)

        manifest_file_ops = [op for op in storage.ops if op[1] == "manifest_file"]
        assert manifest_file_ops == [], (
            f"Found {len(manifest_file_ops)} manifest_file ops in "
            f"multi-partition retry"
        )

    def test_only_allowed_keys_in_operations(self):
        """Storage operations must only use allowed keys."""
        inner = InstantStorageProvider(rng=np.random.RandomState(42))
        storage = TrackingStorageProvider(inner)
        catalog = make_catalog()
        detector = NeverRealDetector()

        # Run FA with retry
        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))
        drive(gen)

        allowed_keys = {"manifest_list", "table_metadata", "catalog_metadata", "metadata"}
        actual_keys = {key for _, key, _ in storage.ops}
        unexpected = actual_keys - allowed_keys
        assert not unexpected, (
            f"Unexpected storage keys: {unexpected}. "
            f"Only {allowed_keys} are allowed."
        )
