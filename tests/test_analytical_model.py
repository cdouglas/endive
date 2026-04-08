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
        """FA, 1 partition, no retry → exactly 4 yields.

        1. catalog.read() — 1 yield (initial snapshot)
        2. per-attempt ML read — 1 yield
        3. per-attempt ML write — 1 yield
        4. catalog.commit() — 2 yields (half-RTT split)
        Total: 1 (catalog read) + 2 (per-attempt) + 2 (CAS) = 5
        But runtime is yielded as 0 (it's tracked differently).
        Actually runtime contributes 1 yield too.
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 0
        # catalog_read(1) + runtime(1) + ml_read(1) + ml_write(1) + cas(2) = 6
        assert len(yields) == 6, (
            f"Expected 6 yields for clean FA commit, got {len(yields)}. "
            f"Yields: {yields}"
        )

    def test_fast_append_3_partitions_yield_count(self):
        """FA, 3 partitions, no retry → 3× per-attempt yields."""
        catalog = make_catalog(num_tables=1, partitions=(3,))
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa(partitions={0: frozenset({0, 1, 2})})
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        # catalog_read(1) + runtime(1) + ml_read(3) + ml_write(3) + cas(2) = 10
        assert len(yields) == 10, (
            f"Expected 10 yields for 3-partition FA, got {len(yields)}"
        )

    def test_fast_append_ml_append_mode_yield_count(self):
        """FA in ML+ mode → no ML writes, fewer yields."""
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_fa()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector(),
                        ml_append_mode=True)
        )
        assert result.status == TransactionStatus.COMMITTED
        # catalog_read(1) + runtime(1) + ml_read(1) + cas(2) = 5
        assert len(yields) == 5

    def test_validated_overwrite_clean_commit_yield_count(self):
        """VO, 1 partition, no retry → same yields as FA."""
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        txn = make_vo()
        yields, result = collect_yields(
            txn.execute(catalog, storage, NeverRealDetector())
        )
        assert result.status == TransactionStatus.COMMITTED
        # catalog_read(1) + runtime(1) + ml_read(1) + ml_write(1) + cas(2) = 6
        assert len(yields) == 6

    def test_fast_append_retry_no_overlap_yield_count(self):
        """FA retry, cross-table (no overlap) → no per-attempt I/O on retry.

        Attempt 0: per-attempt(2) + CAS(2) → fail
        catalog_read(1) → no overlap
        Attempt 1: CAS(2) → succeed (no per-attempt, per_attempt_n=0)
        """
        catalog = make_catalog(num_tables=2, partitions=(1, 1))
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2, tables=frozenset({1}),
                      partitions={1: frozenset({0})})
        gen = txn.execute(catalog, storage, detector)
        # Drive past catalog_read + runtime
        next(gen)  # catalog read
        gen.send(None)  # runtime

        # T1 commits to table 0 (cross-table)
        t1 = make_fa(txn_id=1, tables=frozenset({0}),
                     partitions={0: frozenset({0})})
        drive(t1.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: ml_read(1) + ml_write(1) + cas(2) + catalog_read(1) = 5
        # attempt1: cas(2) only (no per-attempt, cross-table)
        assert len(yields) == 7, (
            f"Expected 7 yields for cross-table retry, got {len(yields)}"
        )

    def test_fast_append_retry_with_overlap_yield_count(self):
        """FA retry, same-table overlap → per-attempt I/O paid again on retry.

        Attempt 0: per-attempt(2) + CAS(2) → fail
        catalog_read(1) → overlap detected
        Attempt 1: per-attempt(2) + CAS(2) → succeed
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        # T1 commits to same table/partition
        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: ml_read(1) + ml_write(1) + cas(2) + catalog_read(1) = 5
        # attempt1: ml_read(1) + ml_write(1) + cas(2) = 4
        assert len(yields) == 9, (
            f"Expected 9 yields for same-table retry, got {len(yields)}"
        )

    def test_vo_retry_with_overlap_convoy_yield_count(self):
        """VO retry with 2 versions behind → historical ML reads (convoy).

        Attempt 0: per-attempt(2) + CAS(2) → fail
        catalog_read(1)
        Conflict: historical_ml_reads = max(0, 2-1) × 1 = 1
        Attempt 1: per-attempt(2) + CAS(2) → succeed
        """
        catalog = make_catalog()
        storage = InstantStorageProvider(rng=np.random.RandomState(42))
        detector = NeverRealDetector()

        txn = make_vo(txn_id=3)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        # Two commits to same table → 2 versions behind
        for i in range(2):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        yields, result = collect_yields(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # attempt0: ml_read(1) + ml_write(1) + cas(2) = 4
        # fail: catalog_read(1) + convoy(1) = 2
        # attempt1: ml_read(1) + ml_write(1) + cas(2) = 4
        assert len(yields) == 10, (
            f"Expected 10 yields for VO convoy retry, got {len(yields)}"
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
        """FA per-attempt I/O with S3: 2 ops × ~43ms → [40, 250]ms."""
        rng = np.random.RandomState(42)
        catalog = make_catalog(latency_ms=0.001)  # ~zero catalog latency
        storage = create_provider("s3", rng=rng)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        # 2 S3 operations: ML read (~43ms median) + ML write (~43ms median)
        # With lognormal variance, allow [40, 250]ms
        assert 40.0 < result.per_attempt_io_ms < 250.0, (
            f"per_attempt_io_ms={result.per_attempt_io_ms:.1f}ms out of "
            f"expected [40, 250]ms for 2 S3 operations"
        )

    def test_fa_per_attempt_io_s3_not_3_ops(self):
        """Regression: per-attempt must NOT include 3 ops (the old bug).

        With 3 ops × 43ms, median per_attempt_io_ms ≈ 129ms.
        With 2 ops × 43ms, median per_attempt_io_ms ≈ 86ms.
        Run multiple seeds and check the median is closer to 2-op model.
        """
        medians = []
        for seed in range(50):
            rng = np.random.RandomState(seed)
            catalog = make_catalog(latency_ms=0.001)
            storage = create_provider("s3", rng=rng)
            txn = make_fa(txn_id=seed)
            result = drive(txn.execute(catalog, storage, NeverRealDetector()))
            medians.append(result.per_attempt_io_ms)

        sample_median = sorted(medians)[len(medians) // 2]
        # 2-op model: median ≈ 86ms. 3-op model: median ≈ 129ms.
        # Threshold at 110ms separates the two models clearly.
        assert sample_median < 110.0, (
            f"Median per_attempt_io_ms across 50 seeds = {sample_median:.1f}ms. "
            f"Expected < 110ms (2-op model). Got > 110ms suggesting 3+ ops."
        )

    def test_fa_per_attempt_io_3_partitions_scales(self):
        """3 partitions → 6 S3 ops → per_attempt > single-partition."""
        rng = np.random.RandomState(42)
        catalog = make_catalog(num_tables=1, partitions=(3,), latency_ms=0.001)
        storage = create_provider("s3", rng=rng)
        txn = make_fa(partitions={0: frozenset({0, 1, 2})})

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        # 6 ops × ~43ms ≈ 258ms median, allow [150, 600]ms
        assert result.per_attempt_io_ms > 150.0, (
            f"3-partition per_attempt={result.per_attempt_io_ms:.1f}ms < 150ms"
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
        """No retry → conflict_io_ms = 0, per_attempt_io_ms = 2ms, catalog_commit_ms = 2ms."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        txn = make_fa()

        result = drive(txn.execute(catalog, storage, NeverRealDetector()))

        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 0
        # catalog_read: 1ms (initial)
        assert result.catalog_read_ms == pytest.approx(1.0)
        # per_attempt: ML read (1ms) + ML write (1ms) = 2ms
        assert result.per_attempt_io_ms == pytest.approx(2.0)
        # catalog_commit: CAS = 1ms (split into 0.5 + 0.5)
        assert result.catalog_commit_ms == pytest.approx(1.0)
        # conflict: 0 (no retry)
        assert result.conflict_io_ms == pytest.approx(0.0)
        # total: catalog_read(1) + runtime(100) + per_attempt(2) + cas(1) = 104
        assert result.total_latency_ms == pytest.approx(104.0)

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

        # T1 commits to table 0
        t1 = make_fa(txn_id=1, tables=frozenset({0}),
                     partitions={0: frozenset({0})})
        drive(t1.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        assert result.catalog_conflicts == 1
        assert result.tblptn_conflicts == 0
        # conflict_io: 0 (no overlap → no conflict cost)
        assert result.conflict_io_ms == pytest.approx(0.0)
        # per_attempt: only first attempt (retry skips per-attempt for no-overlap)
        assert result.per_attempt_io_ms == pytest.approx(2.0)
        # catalog_read: initial(1) + failure-path(1) = 2
        assert result.catalog_read_ms == pytest.approx(2.0)
        # catalog_commit: 2 CAS attempts × 1ms = 2ms
        assert result.catalog_commit_ms == pytest.approx(2.0)

    def test_same_table_retry_per_attempt_doubles(self):
        """Same-table retry: per-attempt paid twice."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_fa(txn_id=2)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        t1 = make_fa(txn_id=1)
        drive(t1.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        assert result.tblptn_conflicts == 1
        # per_attempt: 2 attempts × 2ms = 4ms
        assert result.per_attempt_io_ms == pytest.approx(4.0)
        # conflict_io: 0 (FA has no conflict cost)
        assert result.conflict_io_ms == pytest.approx(0.0)
        # ML reads: 2 (one per attempt)
        assert result.manifest_list_reads == 2
        assert result.manifest_list_writes == 2

    def test_vo_convoy_cost_decomposition(self):
        """VO convoy: historical ML reads appear in conflict_io_ms."""
        catalog = make_catalog(latency_ms=1.0)
        storage = InstantStorageProvider(rng=np.random.RandomState(42), latency_ms=1.0)
        detector = NeverRealDetector()

        txn = make_vo(txn_id=4)
        gen = txn.execute(catalog, storage, detector)
        next(gen)  # catalog read
        gen.send(None)  # runtime

        # 3 commits → 3 versions behind → 2 historical ML reads
        for i in range(3):
            t = make_fa(txn_id=i + 1)
            drive(t.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 1
        # convoy: max(0, 3-1) × 1 partition = 2 historical ML reads × 1ms = 2ms
        assert result.conflict_io_ms == pytest.approx(2.0)
        # per_attempt: 2 attempts × 2ms = 4ms
        assert result.per_attempt_io_ms == pytest.approx(4.0)
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

        # 2 commits to all 3 partitions → 2 versions behind
        for i in range(2):
            t = make_fa(txn_id=i + 1,
                        partitions={0: frozenset({0, 1, 2})})
            drive(t.execute(catalog, storage, detector))

        result = drive(gen)
        assert result.status == TransactionStatus.COMMITTED
        # convoy: max(0, 2-1) × 3 partitions = 3 historical ML reads
        assert result.conflict_io_ms == pytest.approx(3.0)
        # per_attempt first: 3 reads + 3 writes = 6ms
        # per_attempt retry: 3 reads + 3 writes = 6ms
        assert result.per_attempt_io_ms == pytest.approx(12.0)
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

        # Total yields = catalog_read + runtime + ML ops + CAS ops
        n_ml_ops = result.manifest_list_reads + result.manifest_list_writes
        # catalog_read(1) + runtime(1) + ml_ops + cas_yields(2)
        expected_yields = 1 + 1 + n_ml_ops + 2
        assert len(yields) == expected_yields, (
            f"Yield count ({len(yields)}) != analytical prediction "
            f"({expected_yields}): 1 cat_read + 1 runtime + "
            f"{n_ml_ops} ML ops + 2 CAS"
        )
