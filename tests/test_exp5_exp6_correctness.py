"""Integration tests for exp5/exp6 experiment correctness.

Validates that the I/O cost model produces correct results for
partition-aware (exp5) and inlined metadata (exp6) experiments.
"""

import numpy as np
import pytest

from endive.catalog import CASCatalog, InstantCatalog
from endive.conflict_detector import ProbabilisticConflictDetector
from endive.simulation import Simulation, SimulationConfig
from endive.storage import InstantStorageProvider, LognormalLatency, create_provider
from endive.workload import (
    Workload,
    WorkloadConfig,
    UniformPartitionSelector,
)


def make_workload_config(
    num_tables=1,
    partitions_per_table=(32,),
    partitions_per_txn=1,
    fast_append_weight=1.0,
    validated_overwrite_weight=0.0,
    inter_arrival_ms=200.0,
    runtime_ms=50.0,
):
    return WorkloadConfig(
        inter_arrival=LognormalLatency.from_median(
            median_ms=inter_arrival_ms, sigma=0.001,
        ),
        runtime=LognormalLatency.from_median(
            median_ms=runtime_ms, sigma=0.001,
        ),
        num_tables=num_tables,
        partitions_per_table=partitions_per_table,
        fast_append_weight=fast_append_weight,
        validated_overwrite_weight=validated_overwrite_weight,
        partitions_per_txn=partitions_per_txn,
        partition_selector=UniformPartitionSelector(),
    )


# ---------------------------------------------------------------------------
# exp5: Non-inlined, partition-aware
# ---------------------------------------------------------------------------

class TestExp5NonInlinedPartitions:
    """Verify non-inlined partition-aware experiments produce correct I/O."""

    def test_tm_io_present_non_inlined(self):
        """Non-inlined: table_metadata_reads > 0 and table_metadata_writes > 0."""
        wl = make_workload_config(partitions_per_txn=1)
        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            catalog=InstantCatalog(1, (32,), latency_ms=1.0),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42)),
            workload=Workload(wl, seed=42),
            conflict_detector=ProbabilisticConflictDetector(0.0,
                rng=np.random.RandomState(42)),
            max_retries=10,
            metadata_inlined=False,
        )
        stats = Simulation(config).run()
        assert stats.committed > 5
        assert stats.table_metadata_reads > 0
        assert stats.table_metadata_writes > 0
        # Every committed txn should have at least 1 TM read and 1 TM write
        for r in stats.transactions:
            if r.status.name == "COMMITTED":
                assert r.table_metadata_reads >= 1
                assert r.table_metadata_writes >= 1

    def test_ml_scales_with_partitions_per_txn(self):
        """More partitions per txn → more ML reads/writes per committed txn."""
        results = {}
        for ppt in [1, 4]:
            wl = make_workload_config(partitions_per_txn=ppt,
                                      inter_arrival_ms=500.0)
            config = SimulationConfig(
                duration_ms=5000.0,
                seed=42,
                catalog=InstantCatalog(1, (32,), latency_ms=1.0),
                storage_provider=InstantStorageProvider(
                    rng=np.random.RandomState(42)),
                workload=Workload(wl, seed=42),
                conflict_detector=ProbabilisticConflictDetector(0.0,
                    rng=np.random.RandomState(42)),
                max_retries=10,
                metadata_inlined=False,
            )
            stats = Simulation(config).run()
            assert stats.committed > 3
            avg_ml = stats.manifest_list_reads / stats.committed
            results[ppt] = avg_ml

        # ppt=4 should have ~4× the ML reads per txn as ppt=1
        assert results[4] > results[1] * 2, (
            f"ppt=4 ML reads/txn ({results[4]:.1f}) should be >> "
            f"ppt=1 ({results[1]:.1f})"
        )

    def test_conflict_sum_invariant(self):
        """catalog_conflicts + tblptn_conflicts == total_retries for every txn."""
        wl = make_workload_config(partitions_per_txn=1, inter_arrival_ms=100.0)
        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            catalog=InstantCatalog(1, (32,), latency_ms=1.0),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42)),
            workload=Workload(wl, seed=42),
            conflict_detector=ProbabilisticConflictDetector(0.0,
                rng=np.random.RandomState(42)),
            max_retries=10,
            metadata_inlined=False,
        )
        stats = Simulation(config).run()
        for r in stats.transactions:
            assert r.catalog_conflicts + r.tblptn_conflicts == r.total_retries, (
                f"txn {r.txn_id}: {r.catalog_conflicts} + {r.tblptn_conflicts} "
                f"!= {r.total_retries}"
            )

    def test_disjoint_retries_are_free(self):
        """With 32 partitions and ppt=1, most retries should be disjoint (free)."""
        wl = make_workload_config(partitions_per_txn=1, inter_arrival_ms=100.0)
        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            catalog=InstantCatalog(1, (32,), latency_ms=1.0),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42)),
            workload=Workload(wl, seed=42),
            conflict_detector=ProbabilisticConflictDetector(0.0,
                rng=np.random.RandomState(42)),
            max_retries=10,
            metadata_inlined=False,
        )
        stats = Simulation(config).run()
        total_retries = sum(r.total_retries for r in stats.transactions)
        catalog_retries = sum(r.catalog_conflicts for r in stats.transactions)
        if total_retries > 0:
            # With 32 partitions and ppt=1, ~97% of retries should be disjoint
            disjoint_fraction = catalog_retries / total_retries
            assert disjoint_fraction > 0.5, (
                f"Disjoint fraction {disjoint_fraction:.1%} too low "
                f"for 32 partitions with ppt=1"
            )


# ---------------------------------------------------------------------------
# exp6: Inlined metadata
# ---------------------------------------------------------------------------

class TestExp6InlinedMetadata:
    """Verify inlined metadata experiments produce correct I/O."""

    def test_tm_io_absent_inlined(self):
        """Inlined: table_metadata_reads == 0 and table_metadata_writes == 0."""
        wl = make_workload_config(partitions_per_txn=1)
        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            catalog=InstantCatalog(1, (32,), latency_ms=1.0,
                                   metadata_inlined=True,
                                   initial_partition_size_bytes=16000),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42)),
            workload=Workload(wl, seed=42),
            conflict_detector=ProbabilisticConflictDetector(0.0,
                rng=np.random.RandomState(42)),
            max_retries=10,
            metadata_inlined=True,
        )
        stats = Simulation(config).run()
        assert stats.committed > 5
        assert stats.table_metadata_reads == 0
        assert stats.table_metadata_writes == 0

    def test_ml_io_still_present_inlined(self):
        """Inlined: ML reads/writes should still be present (ML is not inlined)."""
        wl = make_workload_config(partitions_per_txn=1)
        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            catalog=InstantCatalog(1, (32,), latency_ms=1.0,
                                   metadata_inlined=True,
                                   initial_partition_size_bytes=16000),
            storage_provider=InstantStorageProvider(
                rng=np.random.RandomState(42)),
            workload=Workload(wl, seed=42),
            conflict_detector=ProbabilisticConflictDetector(0.0,
                rng=np.random.RandomState(42)),
            max_retries=10,
            metadata_inlined=True,
        )
        stats = Simulation(config).run()
        assert stats.manifest_list_reads > 0
        assert stats.manifest_list_writes > 0

    def test_inlined_faster_than_non_inlined(self):
        """Same workload: inlined should have lower per-attempt latency."""
        per_attempt = {}
        for inlined in [False, True]:
            wl = make_workload_config(partitions_per_txn=1,
                                      inter_arrival_ms=500.0)
            cat_kwargs = dict(
                num_tables=1, partitions_per_table=(32,), latency_ms=1.0,
            )
            if inlined:
                cat_kwargs.update(metadata_inlined=True,
                                  initial_partition_size_bytes=16000)
            config = SimulationConfig(
                duration_ms=5000.0,
                seed=42,
                catalog=InstantCatalog(**cat_kwargs),
                storage_provider=InstantStorageProvider(
                    rng=np.random.RandomState(42)),
                workload=Workload(wl, seed=42),
                conflict_detector=ProbabilisticConflictDetector(0.0,
                    rng=np.random.RandomState(42)),
                max_retries=10,
                metadata_inlined=inlined,
            )
            stats = Simulation(config).run()
            committed = [r for r in stats.transactions
                         if r.status.name == "COMMITTED"]
            avg = sum(r.per_attempt_io_ms for r in committed) / len(committed)
            per_attempt[inlined] = avg

        # Inlined should be cheaper (no TM I/O)
        assert per_attempt[True] < per_attempt[False], (
            f"Inlined per_attempt ({per_attempt[True]:.1f}ms) should be < "
            f"non-inlined ({per_attempt[False]:.1f}ms)"
        )

    def test_cas_size_scaling_with_s3(self):
        """With S3, inlined 500 KiB CAS should be more expensive than 100-byte CAS.

        Uses multiple seeds to overcome lognormal variance.
        """
        from endive.transaction import FastAppendTransaction

        class _Det:
            def is_real_conflict(self, txn, c, s):
                return False

        cas_times = {False: [], True: []}
        for seed in range(20):
            for inlined in [False, True]:
                rng = np.random.RandomState(seed)
                storage = create_provider("s3", rng=rng)
                cat_kwargs = dict(
                    storage=storage,
                    num_tables=1, partitions_per_table=(32,),
                )
                if inlined:
                    cat_kwargs.update(metadata_inlined=True,
                                      initial_partition_size_bytes=16000)
                cat = CASCatalog(**cat_kwargs)
                txn = FastAppendTransaction(
                    txn_id=1, submit_time_ms=0.0, runtime_ms=1.0,
                    tables_written=frozenset({0}),
                    partitions_written={0: frozenset({0})},
                )
                gen = txn.execute(cat, storage, _Det(),
                                  metadata_inlined=inlined)
                try:
                    while True:
                        next(gen)
                except StopIteration as e:
                    cas_times[inlined].append(e.value.catalog_commit_ms)

        med_non = sorted(cas_times[False])[10]
        med_inl = sorted(cas_times[True])[10]
        # 500 KiB inlined CAS should have higher median than 100-byte
        assert med_inl > med_non, (
            f"Inlined CAS median ({med_inl:.1f}ms) should be > "
            f"non-inlined ({med_non:.1f}ms)"
        )


# ---------------------------------------------------------------------------
# Cross-cutting: exp5 vs exp6 comparison
# ---------------------------------------------------------------------------

class TestExp5VsExp6Comparison:
    """Verify the key differences between non-inlined and inlined."""

    def test_same_ml_io_different_tm_io(self):
        """Both should have same ML I/O; only TM I/O differs."""
        ml_counts = {}
        tm_counts = {}
        for inlined in [False, True]:
            wl = make_workload_config(partitions_per_txn=1,
                                      inter_arrival_ms=500.0)
            config = SimulationConfig(
                duration_ms=5000.0,
                seed=42,
                catalog=InstantCatalog(1, (32,), latency_ms=1.0,
                                       metadata_inlined=inlined,
                                       initial_partition_size_bytes=16000
                                       if inlined else 2048),
                storage_provider=InstantStorageProvider(
                    rng=np.random.RandomState(42)),
                workload=Workload(wl, seed=42),
                conflict_detector=ProbabilisticConflictDetector(0.0,
                    rng=np.random.RandomState(42)),
                max_retries=10,
                metadata_inlined=inlined,
            )
            stats = Simulation(config).run()
            ml_counts[inlined] = (stats.manifest_list_reads,
                                  stats.manifest_list_writes)
            tm_counts[inlined] = (stats.table_metadata_reads,
                                  stats.table_metadata_writes)

        # ML I/O should be the same (same workload, same seed)
        assert ml_counts[False] == ml_counts[True], (
            f"ML I/O differs: non-inlined={ml_counts[False]}, "
            f"inlined={ml_counts[True]}"
        )
        # TM I/O: non-inlined > 0, inlined == 0
        assert tm_counts[False][0] > 0
        assert tm_counts[True] == (0, 0)
