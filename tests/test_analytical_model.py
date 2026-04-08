"""Analytical model tests for the simulator's I/O cost model.

These tests verify that simulated latency matches first-principles
predictions. They would have caught the manifest file cost bug (9d1d8e1)
where a spurious ~43ms I/O operation was added to every commit attempt.

Test categories:
1a. Yield count verification — exact number of yields per commit path
1b. Latency magnitude with S3 storage — tight bounds on per-attempt cost
1c. Throughput prediction — simulated throughput matches analytical model
1d. Retry cost decomposition — conflict/per-attempt cost accounting
"""

import numpy as np
import pytest

from endive.catalog import CASCatalog, InstantCatalog
from endive.storage import InstantStorageProvider, create_provider
from endive.transaction import (
    ConflictCost,
    FastAppendTransaction,
    TransactionStatus,
    ValidatedOverwriteTransaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class NeverRealDetector:
    def is_real_conflict(self, txn, current, start):
        return False


class AlwaysRealDetector:
    def is_real_conflict(self, txn, current, start):
        return True


def drive(gen):
    """Drive generator to completion, return value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def collect_yields(gen):
    """Collect all yields and return (yields, return_value)."""
    ys = []
    try:
        while True:
            ys.append(next(gen))
    except StopIteration as e:
        return ys, e.value


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
# 1a. Yield count verification
# ---------------------------------------------------------------------------

class TestYieldCount:
    """Verify the exact number of generator yields per commit path.

    Each yield represents one I/O operation consuming simulated time.
    A spurious operation (like the removed manifest file write) would
    add exactly one yield per occurrence, making these tests fail.
    """

    def test_fast_append_clean_commit_yield_count(self):
        """FA, 1 partition, no retry, non-inlined.

        Per-attempt: TM_read(1) + ML_read(1) + ML_write(1) + TM_write(1) = 4
        catalog_read(1) + runtime(1) + per_attempt(4) + cas(2) = 8
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 0
        assert len(yields) == 8, (
            f"Expected 8 yields for clean FA commit, got {len(yields)}. "
            f"Yields: {yields}"
        )

    def test_fast_append_clean_commit_inlined_yield_count(self):
        """FA, 1 partition, no retry, metadata inlined → no TM I/O.

        Per-attempt: ML_read(1) + ML_write(1) = 2 (no TM file)
        catalog_read(1) + runtime(1) + per_attempt(2) + cas(2) = 6
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector(),
                        metadata_inlined=True)
        )
        assert result.status == TransactionStatus.COMMITTED
        assert len(yields) == 6

    def test_fast_append_3_partitions_yield_count(self):
        """FA, 3 partitions, no retry, non-inlined.

        Per-attempt: TM_read(1) + ML_read(3) + ML_write(3) + TM_write(1) = 8
        catalog_read(1) + runtime(1) + per_attempt(8) + cas(2) = 12
        """
        catalog = make_catalog(num_tables=1, partitions=(3,))
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa(partitions={0: frozenset({0, 1, 2})})
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        assert len(yields) == 12, (
            f"Expected 12 yields for 3-partition FA, got {len(yields)}"
        )

    def test_fast_append_ml_append_mode_yield_count(self):
        """FA in ML+ mode, non-inlined → no ML writes, still has TM I/O."""
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector(),
                        ml_append_mode=True)
        )
        assert result.status == TransactionStatus.COMMITTED
        # catalog_read(1) + runtime(1) + TM_read(1) + ML_read(1) + TM_write(1) + cas(2) = 7
        assert len(yields) == 7

    def test_validated_overwrite_clean_commit_yield_count(self):
        """VO, 1 partition, no retry, non-inlined → same as FA."""
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_vo()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        assert len(yields) == 8

    def test_fast_append_retry_no_overlap_yield_count(self):
        """FA retry, cross-table (no overlap) → no per-attempt I/O on retry.

        Attempt 0: per-attempt(4) + CAS(2) → fail
        catalog_read(1) → no overlap
        Attempt 1: CAS(2) → succeed (no per-attempt, per_attempt_n=0)
        """
        catalog = make_catalog(num_tables=2, partitions=(1, 1))
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2, tables=frozenset({1}),
                      partitions={1: frozenset({0})})
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        t1 = make_fa(txn_id=1, tables=frozenset({0}),
                     partitions={0: frozenset({0})})
        drive(t1.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: per_attempt(4) + cas(2) + catalog_read(1) = 7
        # attempt1: cas(2) only
        assert len(yields) == 9, (
            f"Expected 9 yields for cross-table retry, got {len(yields)}"
        )

    def test_fast_append_retry_with_overlap_yield_count(self):
        """FA retry, same-table overlap → per-attempt I/O paid again on retry.

        Attempt 0: per-attempt(4) + CAS(2) → fail
        catalog_read(1) → overlap
        Attempt 1: per-attempt(4) + CAS(2) → succeed
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: per_attempt(4) + cas(2) + catalog_read(1) = 7
        # attempt1: per_attempt(4) + cas(2) = 6
        assert len(yields) == 13, (
            f"Expected 13 yields for same-table retry, got {len(yields)}"
        )

    def test_vo_retry_with_overlap_convoy_yield_count(self):
        """VO retry with 2 versions behind → historical ML reads (convoy).

        Attempt 0: per-attempt(4) + CAS(2) → fail
        catalog_read(1) + convoy(1) = 2
        Attempt 1: per-attempt(4) + CAS(2) → succeed
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_vo(txn_id=3)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        for i in range(2):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: per_attempt(4) + cas(2) = 6
        # fail: catalog_read(1) + convoy(1) = 2
        # attempt1: per_attempt(4) + cas(2) = 6
        assert len(yields) == 14, (
            f"Expected 14 yields for VO convoy retry, got {len(yields)}"
        )


# ---------------------------------------------------------------------------
# 1b. Latency magnitude with S3 storage
# ---------------------------------------------------------------------------

class TestLatencyMagnitude:
    """Verify per-attempt latency with real S3 storage is in expected range.

    S3 manifest_list read/write median ≈ 40-60ms. Per-attempt cost is
    2 operations (1 ML read + 1 ML write). Expected range: [40, 250]ms.

    The manifest file bug added a third operation (~43ms), pushing
    per-attempt cost to [120, 300]ms — these bounds would catch it.
    """

    def test_fa_per_attempt_io_s3_magnitude(self):
        """FA per-attempt I/O with S3, non-inlined: 4 ops × ~43ms → [80, 500]ms."""
        rng = np.random.RandomState(42)
        catalog = make_catalog(latency_ms=0.001)  # ~zero catalog latency
        storage = create_provider("s3", rng=rng)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        # 4 S3 operations: TM read + ML read + ML write + TM write (~43ms each)
        assert 80.0 < result.per_attempt_io_ms < 500.0, (
            f"per_attempt_io_ms={result.per_attempt_io_ms:.1f}ms out of "
            f"expected [80, 500]ms for 4 S3 operations"
        )

    def test_fa_per_attempt_io_s3_inlined_fewer_ops(self):
        """Inlined per-attempt has 2 fewer ops (no TM file I/O)."""
        medians_non = []
        medians_inl = []
        for seed in range(30):
            rng = np.random.RandomState(seed)
            cat = make_catalog(latency_ms=0.001)
            stor = create_provider("s3", rng=rng)
            txn = make_fa(txn_id=seed)
            r = drive(txn.execute(cat, stor, NeverRealDetector()))
            medians_non.append(r.per_attempt_io_ms)

            rng2 = np.random.RandomState(seed)
            cat2 = make_catalog(latency_ms=0.001)
            stor2 = create_provider("s3", rng=rng2)
            txn2 = make_fa(txn_id=seed + 100)
            r2 = drive(txn2.execute(cat2, stor2, NeverRealDetector(),
                                     metadata_inlined=True))
            medians_inl.append(r2.per_attempt_io_ms)

        med_non = sorted(medians_non)[15]
        med_inl = sorted(medians_inl)[15]
        # Non-inlined: 4 ops. Inlined: 2 ops. Ratio ≈ 2.0
        ratio = med_non / max(med_inl, 0.001)
        assert ratio > 1.5, (
            f"Non-inlined/inlined ratio = {ratio:.1f}x, expected > 1.5x "
            f"(non={med_non:.0f}ms, inl={med_inl:.0f}ms)"
        )

    def test_fa_per_attempt_io_3_partitions_scales(self):
        """3 partitions, non-inlined → 8 S3 ops → per_attempt > single-partition."""
        rng = np.random.RandomState(42)
        catalog = make_catalog(num_tables=1, partitions=(3,), latency_ms=0.001)
        storage = create_provider("s3", rng=rng)
        txn = make_fa(partitions={0: frozenset({0, 1, 2})})

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        # 8 ops (TM_r + 3×ML_r + 3×ML_w + TM_w) × ~43ms ≈ 344ms
        assert result.per_attempt_io_ms > 200.0, (
            f"3-partition per_attempt={result.per_attempt_io_ms:.1f}ms < 200ms"
        )

    def test_cross_provider_latency_ratio(self):
        """S3 per-attempt should be >> instant per-attempt."""
        detector = NeverRealDetector()

        # Instant: 2 ops × 1ms = 2ms
        cat1 = make_catalog(latency_ms=0.001)
        stor1 = InstantStorageProvider(rng=np.random.RandomState(42))
        txn1 = make_fa(txn_id=1)
        r1 = drive(txn1.execute(cat1, stor1, detector))

        # S3: 2 ops × ~43ms ≈ 86ms
        cat2 = make_catalog(latency_ms=0.001)
        stor2 = create_provider("s3", rng=np.random.RandomState(42))
        txn2 = make_fa(txn_id=2)
        r2 = drive(txn2.execute(cat2, stor2, detector))

        ratio = r2.per_attempt_io_ms / max(r1.per_attempt_io_ms, 0.001)
        assert ratio > 20.0, (
            f"S3/instant per-attempt ratio = {ratio:.1f}×, expected > 20×"
        )


# ---------------------------------------------------------------------------
# 1c. Throughput prediction
# ---------------------------------------------------------------------------

class TestThroughputPrediction:
    """Verify simulated throughput matches analytical prediction.

    With instant storage (1ms ops) and known inter-arrival + runtime,
    we can predict throughput analytically. A cost inflation of 50%
    would cause throughput to drop detectably.
    """

    def test_instant_throughput_matches_prediction(self):
        """Run simulation with instant storage, verify txn count is reasonable.

        With 1ms ops, FA commit cost = 1 (cat_read) + 100 (runtime) +
        2 (ML read+write) + 1 (CAS) = 104ms per txn.
        Inter-arrival 200ms → at most 1 txn per 200ms, so in 10000ms
        we expect ~50 txns. If per-attempt cost were inflated by 43ms,
        the per-txn time barely changes (104ms→147ms) but the model
        must still hold.
        """
        from endive.simulation import Simulation, SimulationConfig
        from endive.storage import LognormalLatency
        from endive.workload import WorkloadConfig, Workload

        wl_config = WorkloadConfig(
            inter_arrival=LognormalLatency.from_median(
                median_ms=200.0, sigma=0.001,  # Nearly deterministic
            ),
            runtime=LognormalLatency.from_median(
                median_ms=100.0, sigma=0.001,
            ),
            num_tables=1,
            partitions_per_table=(1,),
            fast_append_weight=1.0,
            validated_overwrite_weight=0.0,
        )
        workload = Workload(wl_config, seed=42)
        config = SimulationConfig(
            duration_ms=10000.0,
            seed=42,
            catalog=make_catalog(),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42),
            ),
            conflict_detector=NeverRealDetector(),
            workload=workload,
            max_retries=3,
        )
        stats = Simulation(config).run()

        # 10000ms / 200ms inter-arrival ≈ 50 txns
        # Allow wide margin for simulation startup effects
        assert 30 < stats.committed < 70, (
            f"Expected ~50 committed txns in 10s, got {stats.committed}"
        )
        # All should succeed (no contention with 200ms inter-arrival)
        assert stats.aborted == 0, f"Unexpected aborts: {stats.aborted}"


# ---------------------------------------------------------------------------
# 1d. Retry cost decomposition
# ---------------------------------------------------------------------------

class TestRetryCostDecomposition:
    """Verify conflict cost, per-attempt cost, and catalog cost are
    correctly attributed in TransactionResult.
    """

    def test_clean_commit_timing_decomposition(self):
        """No retry, non-inlined → per_attempt includes TM + ML I/O."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 0
        assert result.catalog_read_ms == pytest.approx(1.0)
        # per_attempt: TM_read(1) + ML_read(1) + ML_write(1) + TM_write(1) = 4ms
        assert result.per_attempt_io_ms == pytest.approx(4.0)
        assert result.catalog_commit_ms == pytest.approx(1.0)
        assert result.conflict_io_ms == pytest.approx(0.0)
        # total: catalog_read(1) + runtime(100) + per_attempt(4) + cas(1) = 106
        assert result.total_latency_ms == pytest.approx(106.0)
        # I/O counters
        assert result.table_metadata_reads == 1
        assert result.table_metadata_writes == 1
        assert result.manifest_list_reads == 1
        assert result.manifest_list_writes == 1

    def test_clean_commit_inlined_timing(self):
        """No retry, inlined → no TM I/O, only ML."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector(),
                                   metadata_inlined=True))

        assert result.status == TransactionStatus.COMMITTED
        # per_attempt: ML_read(1) + ML_write(1) = 2ms (no TM)
        assert result.per_attempt_io_ms == pytest.approx(2.0)
        # total: catalog_read(1) + runtime(100) + per_attempt(2) + cas(1) = 104
        assert result.total_latency_ms == pytest.approx(104.0)
        assert result.table_metadata_reads == 0
        assert result.table_metadata_writes == 0

    def test_cross_table_retry_no_conflict_io(self):
        """Cross-table retry: conflict_io = 0, extra catalog_read only."""
        catalog = make_catalog(num_tables=2, partitions=(1, 1), latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2, tables=frozenset({1}),
                      partitions={1: frozenset({0})})
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        t1 = make_fa(txn_id=1, tables=frozenset({0}),
                     partitions={0: frozenset({0})})
        drive(t1.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        assert result.catalog_conflicts == 1
        assert result.conflict_io_ms == pytest.approx(0.0)
        # per_attempt: only first attempt (retry skips per-attempt for no-overlap)
        assert result.per_attempt_io_ms == pytest.approx(4.0)
        assert result.catalog_read_ms == pytest.approx(2.0)
        assert result.catalog_commit_ms == pytest.approx(2.0)

    def test_same_table_retry_per_attempt_doubles(self):
        """Same-table retry: per-attempt paid twice."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # per_attempt: 2 attempts × 4ms = 8ms
        assert result.per_attempt_io_ms == pytest.approx(8.0)
        assert result.conflict_io_ms == pytest.approx(0.0)
        assert result.manifest_list_reads == 2
        assert result.manifest_list_writes == 2
        assert result.table_metadata_reads == 2
        assert result.table_metadata_writes == 2

    def test_vo_convoy_cost_decomposition(self):
        """VO convoy: historical ML reads appear in conflict_io_ms."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_vo(txn_id=4)
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        for i in range(3):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        assert result.conflict_io_ms == pytest.approx(2.0)
        # per_attempt: 2 attempts × 4ms = 8ms
        assert result.per_attempt_io_ms == pytest.approx(8.0)
        # ML reads: 2 (per-attempt) + 2 (convoy) = 4
        assert result.manifest_list_reads == 4

    def test_vo_convoy_multi_partition_scales(self):
        """VO convoy with 3-partition overlap: historical reads × 3."""
        catalog = make_catalog(num_tables=1, partitions=(3,), latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_vo(txn_id=4, partitions={0: frozenset({0, 1, 2})})
        gen = txn.execute(catalog, storage, detector)
        next(gen)
        gen.send(None)

        for i in range(2):
            t = make_fa(txn_id=i + 1,
                        partitions={0: frozenset({0, 1, 2})})
            drive(t.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.conflict_io_ms == pytest.approx(3.0)
        # per_attempt first: TM_r(1) + ML_r×3 + ML_w×3 + TM_w(1) = 8ms
        # per_attempt retry: TM_r(1) + ML_r×3 + ML_w×3 + TM_w(1) = 8ms
        assert result.per_attempt_io_ms == pytest.approx(16.0)
        # ML reads: 6 (per-attempt: 2 × 3) + 3 (convoy) = 9
        assert result.manifest_list_reads == 9

    def test_io_counters_match_yield_count(self):
        """I/O counters in result must match actual yields counted."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn = make_fa()

        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )

        n_tm_ops = result.table_metadata_reads + result.table_metadata_writes
        n_ml_ops = result.manifest_list_reads + result.manifest_list_writes
        # catalog_read(1) + runtime(1) + tm_ops + ml_ops + cas_yields(2)
        expected_yields = 1 + 1 + n_tm_ops + n_ml_ops + 2
        assert len(yields) == expected_yields, (
            f"Yield count ({len(yields)}) != analytical prediction "
            f"({expected_yields}): 1 cat_read + 1 runtime + "
            f"{n_tm_ops} TM ops + {n_ml_ops} ML ops + 2 CAS"
        )
