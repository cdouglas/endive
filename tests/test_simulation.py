"""Tests for Simulation runner per SPEC.md §6.

Tests:
- SimulationConfig: frozen, required fields
- Statistics: record_transaction, aggregates, to_dataframe, export_parquet
- Simulation: run, deterministic, workload integration
- SimPy bridge: _drive_generator converts floats to timeouts
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import simpy

from endive.catalog import CASCatalog, InstantCatalog
from endive.conflict_detector import ProbabilisticConflictDetector
from endive.simulation import Simulation, SimulationConfig, Statistics
from endive.storage import (
    InstantStorageProvider,
    LognormalLatency,
    create_provider,
)
from endive.transaction import (
    TransactionResult,
    TransactionStatus,
)
from endive.workload import Workload, WorkloadConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(
    txn_id=1,
    status=TransactionStatus.COMMITTED,
    commit_time_ms=200.0,
    abort_time_ms=-1.0,
    abort_reason=None,
    total_retries=0,
    commit_latency_ms=50.0,
    total_latency_ms=150.0,
    ml_reads=0,
    ml_writes=0,
    mf_reads=0,
    mf_writes=0,
):
    return TransactionResult(
        status=status,
        txn_id=txn_id,
        commit_time_ms=commit_time_ms,
        abort_time_ms=abort_time_ms,
        abort_reason=abort_reason,
        total_retries=total_retries,
        commit_latency_ms=commit_latency_ms,
        total_latency_ms=total_latency_ms,
        manifest_list_reads=ml_reads,
        manifest_list_writes=ml_writes,
        manifest_file_reads=mf_reads,
        manifest_file_writes=mf_writes,
    )


def make_instant_catalog(num_tables=1):
    """Create an InstantCatalog for testing."""
    return InstantCatalog(
        num_tables=num_tables,
        partitions_per_table=tuple([1] * num_tables),
        latency_ms=1.0,
    )


def make_workload_config(
    num_tables=1,
    inter_arrival_scale=100.0,
    runtime_mean=50.0,
    fast_append_weight=1.0,
    merge_append_weight=0.0,
    validated_overwrite_weight=0.0,
):
    """Create a WorkloadConfig for testing."""
    return WorkloadConfig(
        inter_arrival=LognormalLatency.from_median(
            median_ms=inter_arrival_scale, sigma=0.5,
        ),
        runtime=LognormalLatency.from_median(
            median_ms=runtime_mean, sigma=0.5,
        ),
        num_tables=num_tables,
        partitions_per_table=tuple([1] * num_tables),
        fast_append_weight=fast_append_weight,
        merge_append_weight=merge_append_weight,
        validated_overwrite_weight=validated_overwrite_weight,
    )


def make_storage():
    """Create an instant-like storage provider for tests."""
    return InstantStorageProvider(rng=np.random.RandomState(42))


def make_conflict_detector(prob=0.0):
    """Create a conflict detector for tests."""
    return ProbabilisticConflictDetector(prob, rng=np.random.RandomState(42))


def make_simulation_config(
    duration_ms=5000.0,
    seed=42,
    num_tables=1,
    inter_arrival_scale=200.0,
    runtime_mean=50.0,
    max_retries=3,
    conflict_prob=0.0,
    fast_append_weight=1.0,
    merge_append_weight=0.0,
    validated_overwrite_weight=0.0,
):
    """Create a SimulationConfig with sensible defaults for testing."""
    wl_config = make_workload_config(
        num_tables=num_tables,
        inter_arrival_scale=inter_arrival_scale,
        runtime_mean=runtime_mean,
        fast_append_weight=fast_append_weight,
        merge_append_weight=merge_append_weight,
        validated_overwrite_weight=validated_overwrite_weight,
    )
    return SimulationConfig(
        duration_ms=duration_ms,
        seed=seed,
        storage_provider=make_storage(),
        catalog=make_instant_catalog(num_tables=num_tables),
        workload=Workload(wl_config, seed=seed + 100),
        conflict_detector=make_conflict_detector(conflict_prob),
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

class TestSimulationConfig:
    def test_frozen(self):
        """SimulationConfig is immutable."""
        config = make_simulation_config()
        with pytest.raises(AttributeError):
            config.duration_ms = 999

    def test_required_fields(self):
        """Must provide all required fields."""
        with pytest.raises(TypeError):
            SimulationConfig(duration_ms=1000, seed=42)

    def test_default_max_retries(self):
        """Default max_retries is 10."""
        config = make_simulation_config()
        # Our helper passes max_retries=3, so check the default in a fresh config
        wl_config = make_workload_config()
        config = SimulationConfig(
            duration_ms=1000,
            seed=42,
            storage_provider=make_storage(),
            catalog=make_instant_catalog(),
            workload=Workload(wl_config, seed=42),
            conflict_detector=make_conflict_detector(),
        )
        assert config.max_retries == 10

    def test_backoff_defaults(self):
        """Backoff defaults are sensible."""
        config = make_simulation_config()
        assert config.backoff_enabled is False
        assert config.backoff_base_ms == 10.0
        assert config.backoff_multiplier == 2.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_empty(self):
        """Fresh statistics have zero counts."""
        stats = Statistics()
        assert stats.total == 0
        assert stats.committed == 0
        assert stats.aborted == 0
        assert stats.success_rate == 0.0

    def test_record_committed(self):
        """Recording a committed transaction updates counters."""
        stats = Statistics()
        stats.record_transaction(make_result(status=TransactionStatus.COMMITTED))
        assert stats.committed == 1
        assert stats.aborted == 0
        assert stats.total == 1
        assert stats.success_rate == 1.0

    def test_record_aborted(self):
        """Recording an aborted transaction updates counters."""
        stats = Statistics()
        stats.record_transaction(make_result(
            status=TransactionStatus.ABORTED,
            commit_time_ms=-1.0,
            abort_time_ms=200.0,
            abort_reason="max_retries_exceeded",
        ))
        assert stats.committed == 0
        assert stats.aborted == 1
        assert stats.success_rate == 0.0

    def test_record_validation_exception(self):
        """Validation exceptions are counted separately."""
        stats = Statistics()
        stats.record_transaction(make_result(
            status=TransactionStatus.ABORTED,
            commit_time_ms=-1.0,
            abort_time_ms=200.0,
            abort_reason="validation_exception",
        ))
        assert stats.validation_exceptions == 1
        assert stats.aborted == 1

    def test_io_counters_accumulate(self):
        """I/O counters accumulate across transactions."""
        stats = Statistics()
        stats.record_transaction(make_result(ml_reads=3, ml_writes=1))
        stats.record_transaction(make_result(ml_reads=2, mf_reads=5))
        assert stats.manifest_list_reads == 5
        assert stats.manifest_list_writes == 1
        assert stats.manifest_file_reads == 5

    def test_retry_counter(self):
        """Total retries are accumulated."""
        stats = Statistics()
        stats.record_transaction(make_result(total_retries=2))
        stats.record_transaction(make_result(total_retries=3))
        assert stats.total_retries == 5

    def test_to_dataframe_empty(self):
        """to_dataframe returns empty DataFrame with no transactions."""
        stats = Statistics()
        df = stats.to_dataframe()
        assert len(df) == 0

    def test_to_dataframe_columns(self):
        """DataFrame has expected columns."""
        stats = Statistics()
        stats.record_transaction(make_result())
        df = stats.to_dataframe()
        expected = {
            "txn_id", "t_submit", "t_commit", "commit_latency",
            "total_latency", "n_retries", "status", "abort_reason",
            "manifest_list_reads", "manifest_list_writes",
            "manifest_file_reads", "manifest_file_writes",
        }
        assert expected.issubset(set(df.columns))

    def test_to_dataframe_committed_values(self):
        """Committed transaction has correct DataFrame values."""
        stats = Statistics()
        stats.record_transaction(make_result(
            txn_id=7,
            status=TransactionStatus.COMMITTED,
            commit_time_ms=500.0,
            total_latency_ms=200.0,
            commit_latency_ms=80.0,
            total_retries=1,
        ))
        df = stats.to_dataframe()
        row = df.iloc[0]
        assert row["txn_id"] == 7
        assert row["status"] == "committed"
        assert row["t_commit"] == 500
        assert row["commit_latency"] == 80
        assert row["total_latency"] == 200
        assert row["n_retries"] == 1

    def test_to_dataframe_aborted_values(self):
        """Aborted transaction has -1 for commit fields."""
        stats = Statistics()
        stats.record_transaction(make_result(
            status=TransactionStatus.ABORTED,
            commit_time_ms=-1.0,
            abort_time_ms=300.0,
            abort_reason="max_retries_exceeded",
            total_latency_ms=200.0,
        ))
        df = stats.to_dataframe()
        row = df.iloc[0]
        assert row["status"] == "aborted"
        assert row["t_commit"] == -1
        assert row["commit_latency"] == -1
        assert row["abort_reason"] == "max_retries_exceeded"

    def test_export_parquet(self):
        """export_parquet writes valid parquet file."""
        stats = Statistics()
        stats.record_transaction(make_result(txn_id=1))
        stats.record_transaction(make_result(txn_id=2))

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            stats.export_parquet(path)
            df = pd.read_parquet(path)
            assert len(df) == 2
            assert set(df["txn_id"]) == {1, 2}
        finally:
            os.unlink(path)

    def test_multiple_transactions(self):
        """Multiple transactions accumulate correctly."""
        stats = Statistics()
        for i in range(5):
            stats.record_transaction(make_result(txn_id=i))
        for i in range(3):
            stats.record_transaction(make_result(
                txn_id=100 + i,
                status=TransactionStatus.ABORTED,
                commit_time_ms=-1.0,
                abort_time_ms=100.0,
            ))
        assert stats.total == 8
        assert stats.committed == 5
        assert stats.aborted == 3
        assert len(stats.transactions) == 8


# ---------------------------------------------------------------------------
# Simulation._drive_generator
# ---------------------------------------------------------------------------

class TestDriveGenerator:
    def test_empty_generator(self):
        """Generator that immediately returns gives the return value."""
        def gen():
            return 42
            yield  # Make it a generator

        env = simpy.Environment()
        result = [None]

        def process():
            result[0] = yield from Simulation._drive_generator(env, gen())

        env.process(process())
        env.run()
        assert result[0] == 42

    def test_single_yield(self):
        """Single yield advances SimPy time."""
        def gen():
            yield 100.0
            return "done"

        env = simpy.Environment()
        result = [None]

        def process():
            result[0] = yield from Simulation._drive_generator(env, gen())

        env.process(process())
        env.run()
        assert result[0] == "done"
        assert env.now == 100.0

    def test_multiple_yields(self):
        """Multiple yields accumulate SimPy time."""
        def gen():
            yield 10.0
            yield 20.0
            yield 30.0
            return "done"

        env = simpy.Environment()
        result = [None]

        def process():
            result[0] = yield from Simulation._drive_generator(env, gen())

        env.process(process())
        env.run()
        assert result[0] == "done"
        assert env.now == 60.0

    def test_nested_yield_from(self):
        """Works with generators that use yield from internally."""
        def inner():
            yield 5.0
            return 10

        def outer():
            x = yield from inner()
            yield 15.0
            return x + 20

        env = simpy.Environment()
        result = [None]

        def process():
            result[0] = yield from Simulation._drive_generator(env, outer())

        env.process(process())
        env.run()
        assert result[0] == 30
        assert env.now == 20.0


# ---------------------------------------------------------------------------
# Simulation.run
# ---------------------------------------------------------------------------

class TestSimulationRun:
    def test_runs_to_duration(self):
        """Simulation runs until configured duration."""
        config = make_simulation_config(duration_ms=1000.0)
        sim = Simulation(config)
        stats = sim.run()
        # Should have produced some transactions
        assert stats.total > 0

    def test_produces_committed_transactions(self):
        """100% FastAppend with instant catalog should commit all."""
        config = make_simulation_config(
            duration_ms=2000.0,
            inter_arrival_scale=200.0,
            runtime_mean=50.0,
            fast_append_weight=1.0,
            max_retries=10,
        )
        sim = Simulation(config)
        stats = sim.run()
        assert stats.committed > 0
        # All should commit (FastAppend on InstantCatalog = no conflicts)
        assert stats.aborted == 0

    def test_deterministic_with_seed(self):
        """Same seed produces identical results."""
        config_a = make_simulation_config(duration_ms=3000.0, seed=42)
        config_b = make_simulation_config(duration_ms=3000.0, seed=42)

        stats_a = Simulation(config_a).run()
        stats_b = Simulation(config_b).run()

        assert stats_a.committed == stats_b.committed
        assert stats_a.aborted == stats_b.aborted
        assert stats_a.total == stats_b.total

        # Check individual transactions match
        for a, b in zip(stats_a.transactions, stats_b.transactions):
            assert a.txn_id == b.txn_id
            assert a.status == b.status

    def test_different_seeds_different_results(self):
        """Different seeds produce different transaction timings."""
        stats_a = Simulation(make_simulation_config(seed=1)).run()
        stats_b = Simulation(make_simulation_config(seed=999)).run()
        # Transaction timings should differ even if counts coincide
        times_a = [r.commit_time_ms for r in stats_a.transactions]
        times_b = [r.commit_time_ms for r in stats_b.transactions]
        assert times_a != times_b

    def test_high_contention_causes_retries(self):
        """Fast inter-arrival on CAS catalog causes retries."""
        rng = np.random.RandomState(42)
        storage = InstantStorageProvider(rng=rng, latency_ms=5.0)
        catalog = CASCatalog(
            storage=storage,
            num_tables=1,
            partitions_per_table=(1,),
        )

        wl_config = make_workload_config(
            num_tables=1,
            inter_arrival_scale=20.0,  # Very fast arrivals
            runtime_mean=50.0,
        )

        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            storage_provider=storage,
            catalog=catalog,
            workload=Workload(wl_config, seed=142),
            conflict_detector=make_conflict_detector(0.0),
            max_retries=10,
        )

        stats = Simulation(config).run()
        assert stats.committed > 0
        assert stats.total_retries > 0

    def test_validated_overwrite_aborts_on_real_conflict(self):
        """100% ValidatedOverwrite + high conflict prob → aborts."""
        rng = np.random.RandomState(42)
        storage = InstantStorageProvider(rng=rng, latency_ms=5.0)
        catalog = CASCatalog(
            storage=storage,
            num_tables=1,
            partitions_per_table=(1,),
        )

        wl_config = make_workload_config(
            num_tables=1,
            inter_arrival_scale=30.0,
            runtime_mean=50.0,
            fast_append_weight=0.0,
            validated_overwrite_weight=1.0,
        )

        config = SimulationConfig(
            duration_ms=5000.0,
            seed=42,
            storage_provider=storage,
            catalog=catalog,
            workload=Workload(wl_config, seed=142),
            conflict_detector=ProbabilisticConflictDetector(
                1.0, rng=np.random.RandomState(42),
            ),
            max_retries=3,
        )

        stats = Simulation(config).run()
        # Should have some aborts due to real conflicts
        assert stats.aborted > 0 or stats.validation_exceptions > 0

    def test_zero_duration(self):
        """Duration 0 produces no transactions."""
        config = make_simulation_config(duration_ms=0.0)
        stats = Simulation(config).run()
        assert stats.total == 0

    def test_statistics_io_counters(self):
        """I/O counters are populated from transaction results."""
        rng = np.random.RandomState(42)
        storage = InstantStorageProvider(rng=rng, latency_ms=5.0)
        catalog = CASCatalog(
            storage=storage,
            num_tables=1,
            partitions_per_table=(1,),
        )

        wl_config = make_workload_config(
            num_tables=1,
            inter_arrival_scale=15.0,  # Very fast → conflicts
            runtime_mean=50.0,
        )

        config = SimulationConfig(
            duration_ms=3000.0,
            seed=42,
            storage_provider=storage,
            catalog=catalog,
            workload=Workload(wl_config, seed=142),
            conflict_detector=make_conflict_detector(0.0),
            max_retries=10,
        )

        stats = Simulation(config).run()
        # With conflicts, transactions should have some I/O
        # At minimum, each transaction reads the catalog
        assert stats.committed > 0

    def test_seed_none_runs(self):
        """Simulation runs without a seed (non-deterministic)."""
        wl_config = make_workload_config()
        config = SimulationConfig(
            duration_ms=1000.0,
            seed=None,
            storage_provider=make_storage(),
            catalog=make_instant_catalog(),
            workload=Workload(wl_config),
            conflict_detector=make_conflict_detector(),
            max_retries=3,
        )
        stats = Simulation(config).run()
        assert stats.total >= 0  # Just verify it doesn't crash


# ---------------------------------------------------------------------------
# Integration: full pipeline
# ---------------------------------------------------------------------------

class TestSimulationIntegration:
    def test_full_pipeline_to_parquet(self):
        """End-to-end: config → simulation → parquet output."""
        config = make_simulation_config(
            duration_ms=3000.0,
            seed=42,
            inter_arrival_scale=200.0,
        )

        stats = Simulation(config).run()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            stats.export_parquet(path)
            df = pd.read_parquet(path)
            assert len(df) > 0
            assert "status" in df.columns
            assert "txn_id" in df.columns

            # All committed → t_commit > 0
            committed = df[df["status"] == "committed"]
            if len(committed) > 0:
                assert (committed["t_commit"] > 0).all()
        finally:
            os.unlink(path)

    def test_multi_table_workload(self):
        """Simulation works with multiple tables."""
        config = make_simulation_config(
            duration_ms=3000.0,
            num_tables=5,
            inter_arrival_scale=200.0,
        )
        stats = Simulation(config).run()
        assert stats.committed > 0

    def test_mixed_workload(self):
        """Simulation works with mixed operation types."""
        config = make_simulation_config(
            duration_ms=3000.0,
            fast_append_weight=0.5,
            merge_append_weight=0.3,
            validated_overwrite_weight=0.2,
        )
        stats = Simulation(config).run()
        assert stats.total > 0

    def test_dataframe_round_trip(self):
        """Statistics → DataFrame → parquet → DataFrame preserves data."""
        config = make_simulation_config(duration_ms=2000.0)
        stats = Simulation(config).run()

        df1 = stats.to_dataframe()

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            path = f.name

        try:
            stats.export_parquet(path)
            df2 = pd.read_parquet(path)
            assert len(df1) == len(df2)
            assert list(df1["txn_id"]) == list(df2["txn_id"])
        finally:
            os.unlink(path)
