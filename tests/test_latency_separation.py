"""Integration tests for catalog/storage latency separation.

Ensures that catalog and storage latencies are independently configurable
and that per-attempt I/O cost is always paid (even on clean commits).
"""

import tomllib

import numpy as np
import pytest

from endive.catalog import CASCatalog, InstantCatalog
from endive.config import load_simulation_config
from endive.storage import InstantStorageProvider, create_provider
from endive.transaction import (
    FastAppendTransaction,
    TransactionStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NeverRealConflictDetector:
    """Conflict detector that never reports real conflicts."""
    def is_real_conflict(self, txn, current, start):
        return False


def _drive(gen):
    """Drive generator to completion, return its value."""
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def _collect_yields(gen):
    """Drive generator, collect yields and return value."""
    yields = []
    try:
        while True:
            yields.append(next(gen))
    except StopIteration as e:
        return yields, e.value


def _make_fa(txn_id=1, runtime=100.0):
    return FastAppendTransaction(
        txn_id=txn_id,
        submit_time_ms=0.0,
        runtime_ms=runtime,
        tables_written=frozenset({0}),
    )


# ---------------------------------------------------------------------------
# Test: S3 storage + instant catalog → commit includes real storage I/O
# ---------------------------------------------------------------------------

class TestLatencySeparation:
    def test_s3_storage_with_instant_catalog_commit_includes_storage_io(self):
        """With S3 storage and instant catalog, commit latency > 50ms.

        Per-attempt I/O is ML read + MF write + ML write on S3 (~43ms each),
        plus 1ms catalog CAS. Total per attempt: ~130ms minimum.
        """
        rng = np.random.RandomState(42)
        catalog = InstantCatalog(
            num_tables=1,
            partitions_per_table=(1,),
            latency_ms=1.0,
        )
        storage = create_provider("s3", rng=rng)
        detector = _NeverRealConflictDetector()
        txn = _make_fa()

        result = _drive(txn.execute(catalog, storage, detector))

        assert result.status == TransactionStatus.COMMITTED
        # S3 storage I/O: 3 operations × ~43ms each ≈ 129ms minimum
        # Plus 1ms catalog CAS = ~130ms
        assert result.commit_latency_ms > 50.0, (
            f"commit_latency_ms={result.commit_latency_ms:.1f}ms is too low — "
            f"S3 storage I/O should dominate"
        )

    def test_instant_storage_with_instant_catalog_commit_is_fast(self):
        """With both instant, commit latency < 10ms.

        Per-attempt I/O: 3 × 1ms + 1ms CAS = 4ms.
        """
        rng = np.random.RandomState(42)
        catalog = InstantCatalog(
            num_tables=1,
            partitions_per_table=(1,),
            latency_ms=1.0,
        )
        storage = InstantStorageProvider(rng=rng)
        detector = _NeverRealConflictDetector()
        txn = _make_fa()

        result = _drive(txn.execute(catalog, storage, detector))

        assert result.status == TransactionStatus.COMMITTED
        assert result.commit_latency_ms < 10.0, (
            f"commit_latency_ms={result.commit_latency_ms:.1f}ms is too high — "
            f"should be ~4ms with instant storage+catalog"
        )

    def test_changing_storage_provider_changes_commit_cost_not_catalog(self):
        """S3 commit >> instant commit, holding catalog constant."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        detector = _NeverRealConflictDetector()

        # Both use instant catalog
        catalog1 = InstantCatalog(num_tables=1, partitions_per_table=(1,), latency_ms=1.0)
        catalog2 = InstantCatalog(num_tables=1, partitions_per_table=(1,), latency_ms=1.0)

        # Different storage providers
        instant_storage = InstantStorageProvider(rng=rng1)
        s3_storage = create_provider("s3", rng=rng2)

        txn1 = _make_fa(txn_id=1)
        r1 = _drive(txn1.execute(catalog1, instant_storage, detector))

        txn2 = _make_fa(txn_id=2)
        r2 = _drive(txn2.execute(catalog2, s3_storage, detector))

        assert r1.status == TransactionStatus.COMMITTED
        assert r2.status == TransactionStatus.COMMITTED
        # S3 commit should be at least 10× slower than instant
        assert r2.commit_latency_ms > r1.commit_latency_ms * 10, (
            f"S3 commit ({r2.commit_latency_ms:.1f}ms) should be >> "
            f"instant commit ({r1.commit_latency_ms:.1f}ms)"
        )

    def test_per_attempt_io_counters_nonzero_on_success(self):
        """Even on clean commit (no retry), I/O counters are nonzero."""
        rng = np.random.RandomState(42)
        catalog = InstantCatalog(num_tables=1, partitions_per_table=(1,), latency_ms=1.0)
        storage = InstantStorageProvider(rng=rng)
        detector = _NeverRealConflictDetector()
        txn = _make_fa()

        result = _drive(txn.execute(catalog, storage, detector))

        assert result.status == TransactionStatus.COMMITTED
        assert result.total_retries == 0
        # Per-attempt cost must be visible even on first attempt
        assert result.manifest_list_reads >= 1, "Must read ML before CAS"
        assert result.manifest_file_writes >= 1, "Must write MF before CAS"
        assert result.manifest_list_writes >= 1, "Must write ML before CAS"


# ---------------------------------------------------------------------------
# Test: config loading separates catalog and storage
# ---------------------------------------------------------------------------

class TestConfigSeparation:
    def test_config_loading_separates_catalog_and_storage(self):
        """Loading exp1 config produces InstantCatalog + non-instant storage."""
        import tempfile
        import os

        config_content = """
[simulation]
duration_ms = 10000
output_path = "test_results.parquet"
seed = 42

[catalog]
num_tables = 1
num_groups = 1
table_metadata_inlined = true
backend = "service"

[catalog.service]
provider = "instant"
latency_ms = 1.0

[transaction]
retry = 10
runtime.min = 30000
runtime.mean = 180000
runtime.sigma = 1.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0
operation_types.fast_append = 1.0
real_conflict_probability = 0.0
manifest_list_mode = "rewrite"

[storage]
provider = "s3"
max_parallel = 4
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = load_simulation_config(config_path)
            # Catalog should be instant (1ms)
            assert isinstance(config.catalog, InstantCatalog), (
                f"Expected InstantCatalog, got {type(config.catalog).__name__}"
            )
            # Storage should NOT be instant
            assert not isinstance(config.storage_provider, InstantStorageProvider), (
                f"Expected non-instant storage, got {type(config.storage_provider).__name__}"
            )
        finally:
            os.unlink(config_path)

    def test_exp1_config_uses_s3_storage_not_instant(self):
        """Direct TOML check: exp1 config has storage=s3 and catalog.backend=service."""
        config_path = "experiment_configs/exp1_fastappend_baseline.toml"
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        # Storage must be S3, not instant
        assert config["storage"]["provider"] == "s3", (
            f"storage.provider should be 's3', got '{config['storage']['provider']}'"
        )

        # Catalog should use service backend
        assert config["catalog"]["backend"] == "service", (
            f"catalog.backend should be 'service', got '{config['catalog'].get('backend')}'"
        )

        # Catalog service should be instant
        assert config["catalog"]["service"]["provider"] == "instant"
        assert config["catalog"]["service"]["latency_ms"] == 1.0
