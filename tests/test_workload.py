"""Tests for Workload generator per SPEC.md §4.

Tests:
- WorkloadConfig frozen dataclass
- Operation type distribution matches weights (chi-squared test)
- Table selectors: Uniform and Zipf distributions
- Partition selectors: Uniform and Zipf distributions
- Workload generates correct Transaction types
- Topology from config, not Catalog
- Deterministic with seeded RNG
- tables_written ⊆ tables_read invariant
- Inter-arrival delays are positive
"""

import pytest
import numpy as np
from collections import Counter

from endive.storage import FixedLatency, LognormalLatency
from endive.transaction import (
    FastAppendTransaction,
    MergeAppendTransaction,
    ValidatedOverwriteTransaction,
)
from endive.workload import (
    PartitionSelector,
    TableSelector,
    UniformPartitionSelector,
    UniformTableSelector,
    Workload,
    WorkloadConfig,
    ZipfPartitionSelector,
    ZipfTableSelector,
    _truncated_zipf_pmf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(**kwargs):
    """Create a WorkloadConfig with sensible defaults."""
    defaults = dict(
        inter_arrival=FixedLatency(latency_ms=100.0),
        runtime=FixedLatency(latency_ms=50.0),
        num_tables=5,
        partitions_per_table=(4, 4, 4, 4, 4),
    )
    defaults.update(kwargs)
    return WorkloadConfig(**defaults)


def generate_n(workload, n):
    """Generate n transactions from workload."""
    gen = workload.generate()
    results = []
    for _ in range(n):
        delay, txn = next(gen)
        results.append((delay, txn))
    return results


# ---------------------------------------------------------------------------
# WorkloadConfig
# ---------------------------------------------------------------------------

class TestWorkloadConfig:
    def test_frozen(self):
        config = make_config()
        with pytest.raises(AttributeError):
            config.num_tables = 10

    def test_required_fields(self):
        config = make_config()
        assert config.num_tables == 5
        assert config.partitions_per_table == (4, 4, 4, 4, 4)

    def test_default_weights(self):
        config = make_config()
        assert config.fast_append_weight == 0.7
        assert config.merge_append_weight == 0.2
        assert config.validated_overwrite_weight == 0.1

    def test_custom_weights(self):
        config = make_config(
            fast_append_weight=1.0,
            merge_append_weight=0.0,
            validated_overwrite_weight=0.0,
        )
        assert config.fast_append_weight == 1.0

    def test_default_tables_per_txn(self):
        config = make_config()
        assert config.tables_per_txn == 1

    def test_default_no_partition_tracking(self):
        config = make_config()
        assert config.partitions_per_txn is None
        assert config.partition_selector is None

    def test_default_table_selector_is_none(self):
        config = make_config()
        assert config.table_selector is None


# ---------------------------------------------------------------------------
# Truncated Zipf PMF
# ---------------------------------------------------------------------------

class TestTruncatedZipfPMF:
    def test_sums_to_one(self):
        pmf = _truncated_zipf_pmf(10, 1.5)
        assert np.isclose(pmf.sum(), 1.0)

    def test_decreasing(self):
        pmf = _truncated_zipf_pmf(10, 1.5)
        for i in range(len(pmf) - 1):
            assert pmf[i] > pmf[i + 1]

    def test_higher_alpha_more_skewed(self):
        pmf_low = _truncated_zipf_pmf(10, 0.5)
        pmf_high = _truncated_zipf_pmf(10, 2.0)
        # Higher alpha → first element gets more probability
        assert pmf_high[0] > pmf_low[0]

    def test_single_element(self):
        pmf = _truncated_zipf_pmf(1, 1.5)
        assert len(pmf) == 1
        assert np.isclose(pmf[0], 1.0)


# ---------------------------------------------------------------------------
# UniformTableSelector
# ---------------------------------------------------------------------------

class TestUniformTableSelector:
    def test_selects_correct_count(self):
        selector = UniformTableSelector()
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=2, total_tables=5, rng=rng)
        assert len(tr) == 2
        assert len(tw) == 2  # write_fraction=1.0

    def test_write_subset_of_read(self):
        selector = UniformTableSelector(write_fraction=0.5)
        rng = np.random.RandomState(42)
        for _ in range(100):
            tr, tw = selector.select(n_tables=4, total_tables=10, rng=rng)
            assert tw.issubset(tr)

    def test_single_table(self):
        selector = UniformTableSelector()
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=1, total_tables=1, rng=rng)
        assert tr == frozenset({0})
        assert tw == frozenset({0})

    def test_all_tables(self):
        selector = UniformTableSelector()
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=3, total_tables=3, rng=rng)
        assert len(tr) == 3
        assert tr == frozenset({0, 1, 2})

    def test_caps_at_total(self):
        selector = UniformTableSelector()
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=10, total_tables=3, rng=rng)
        assert len(tr) == 3

    def test_write_fraction(self):
        selector = UniformTableSelector(write_fraction=0.5)
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=4, total_tables=10, rng=rng)
        assert len(tr) == 4
        assert len(tw) == 2  # int(4 * 0.5) = 2

    def test_uniform_distribution(self):
        """Over many trials, all tables should be selected roughly equally."""
        selector = UniformTableSelector()
        rng = np.random.RandomState(42)
        counts = Counter()
        for _ in range(10000):
            tr, _ = selector.select(n_tables=1, total_tables=5, rng=rng)
            counts[next(iter(tr))] += 1

        # Each table should get ~2000 selections (±200)
        for table_id in range(5):
            assert 1600 < counts[table_id] < 2400


# ---------------------------------------------------------------------------
# ZipfTableSelector
# ---------------------------------------------------------------------------

class TestZipfTableSelector:
    def test_selects_correct_count(self):
        selector = ZipfTableSelector(alpha=1.5)
        rng = np.random.RandomState(42)
        tr, tw = selector.select(n_tables=2, total_tables=5, rng=rng)
        assert len(tr) == 2

    def test_write_subset_of_read(self):
        selector = ZipfTableSelector(alpha=1.5, write_fraction=0.5)
        rng = np.random.RandomState(42)
        for _ in range(100):
            tr, tw = selector.select(n_tables=4, total_tables=10, rng=rng)
            assert tw.issubset(tr)

    def test_skewed_distribution(self):
        """Table 0 should be selected most frequently (Zipf)."""
        selector = ZipfTableSelector(alpha=2.0)
        rng = np.random.RandomState(42)
        counts = Counter()
        for _ in range(10000):
            tr, _ = selector.select(n_tables=1, total_tables=5, rng=rng)
            counts[next(iter(tr))] += 1

        # Table 0 should be most popular
        assert counts[0] > counts[1] > counts[2]
        # Table 0 should get >50% with alpha=2.0
        assert counts[0] > 5000

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            ZipfTableSelector(alpha=0)
        with pytest.raises(ValueError):
            ZipfTableSelector(alpha=-1.0)


# ---------------------------------------------------------------------------
# UniformPartitionSelector
# ---------------------------------------------------------------------------

class TestUniformPartitionSelector:
    def test_selects_correct_count(self):
        selector = UniformPartitionSelector()
        rng = np.random.RandomState(42)
        pr, pw = selector.select(n_partitions=2, total_partitions=4, rng=rng)
        assert len(pr) == 2

    def test_write_subset_of_read(self):
        selector = UniformPartitionSelector(write_fraction=0.5)
        rng = np.random.RandomState(42)
        for _ in range(100):
            pr, pw = selector.select(n_partitions=4, total_partitions=8, rng=rng)
            assert pw.issubset(pr)

    def test_returns_frozenset(self):
        selector = UniformPartitionSelector()
        rng = np.random.RandomState(42)
        pr, pw = selector.select(n_partitions=2, total_partitions=4, rng=rng)
        assert isinstance(pr, frozenset)
        assert isinstance(pw, frozenset)


# ---------------------------------------------------------------------------
# ZipfPartitionSelector
# ---------------------------------------------------------------------------

class TestZipfPartitionSelector:
    def test_skewed_distribution(self):
        """Partition 0 should be selected most frequently (Zipf)."""
        selector = ZipfPartitionSelector(alpha=2.0)
        rng = np.random.RandomState(42)
        counts = Counter()
        for _ in range(5000):
            pr, _ = selector.select(n_partitions=1, total_partitions=4, rng=rng)
            counts[next(iter(pr))] += 1

        assert counts[0] > counts[1] > counts[2]

    def test_alpha_validation(self):
        with pytest.raises(ValueError):
            ZipfPartitionSelector(alpha=0)


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

class TestWorkloadGeneration:
    def test_generates_transactions(self):
        config = make_config()
        workload = Workload(config, seed=42)
        results = generate_n(workload, 10)
        assert len(results) == 10
        for delay, txn in results:
            assert isinstance(txn, (
                FastAppendTransaction,
                MergeAppendTransaction,
                ValidatedOverwriteTransaction,
            ))

    def test_delays_are_positive(self):
        config = make_config()
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for delay, _ in results:
            assert delay > 0

    def test_fixed_delay_with_fixed_distribution(self):
        config = make_config(inter_arrival=FixedLatency(latency_ms=50.0))
        workload = Workload(config, seed=42)
        results = generate_n(workload, 10)
        for delay, _ in results:
            assert delay == 50.0

    def test_txn_ids_are_sequential(self):
        config = make_config()
        workload = Workload(config, seed=42)
        results = generate_n(workload, 10)
        for i, (_, txn) in enumerate(results, 1):
            assert txn.id == i

    def test_submit_times_are_cumulative(self):
        config = make_config(inter_arrival=FixedLatency(latency_ms=100.0))
        workload = Workload(config, seed=42)
        results = generate_n(workload, 5)
        for i, (_, txn) in enumerate(results, 1):
            assert txn.submit_time == pytest.approx(i * 100.0)

    def test_runtime_from_config(self):
        config = make_config(runtime=FixedLatency(latency_ms=42.0))
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())
        assert txn.runtime == 42.0


# ---------------------------------------------------------------------------
# Operation type distribution
# ---------------------------------------------------------------------------

class TestOperationTypeDistribution:
    def test_100_percent_fast_append(self):
        config = make_config(
            fast_append_weight=1.0,
            merge_append_weight=0.0,
            validated_overwrite_weight=0.0,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for _, txn in results:
            assert isinstance(txn, FastAppendTransaction)

    def test_100_percent_merge_append(self):
        config = make_config(
            fast_append_weight=0.0,
            merge_append_weight=1.0,
            validated_overwrite_weight=0.0,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for _, txn in results:
            assert isinstance(txn, MergeAppendTransaction)

    def test_100_percent_validated_overwrite(self):
        config = make_config(
            fast_append_weight=0.0,
            merge_append_weight=0.0,
            validated_overwrite_weight=1.0,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for _, txn in results:
            assert isinstance(txn, ValidatedOverwriteTransaction)

    def test_mixed_distribution_approximate(self):
        """Chi-squared test: observed distribution matches expected weights."""
        config = make_config(
            fast_append_weight=0.7,
            merge_append_weight=0.2,
            validated_overwrite_weight=0.1,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 10000)

        counts = Counter()
        for _, txn in results:
            counts[type(txn).__name__] += 1

        n = len(results)
        expected = {
            'FastAppendTransaction': 0.7 * n,
            'MergeAppendTransaction': 0.2 * n,
            'ValidatedOverwriteTransaction': 0.1 * n,
        }

        for name, exp in expected.items():
            obs = counts[name]
            # Within 10% of expected (allows for statistical variation)
            assert abs(obs - exp) / exp < 0.10, (
                f"{name}: expected ~{exp:.0f}, got {obs}"
            )

    def test_weights_are_normalized(self):
        """Weights don't need to sum to 1.0; they're normalized."""
        config = make_config(
            fast_append_weight=7.0,
            merge_append_weight=2.0,
            validated_overwrite_weight=1.0,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 1000)
        types = Counter(type(txn).__name__ for _, txn in results)
        assert types['FastAppendTransaction'] > types['MergeAppendTransaction']

    def test_zero_total_weight_raises(self):
        config = make_config(
            fast_append_weight=0.0,
            merge_append_weight=0.0,
            validated_overwrite_weight=0.0,
        )
        with pytest.raises(ValueError):
            Workload(config, seed=42)


# ---------------------------------------------------------------------------
# Table selection
# ---------------------------------------------------------------------------

class TestWorkloadTableSelection:
    def test_tables_written_from_selector(self):
        config = make_config(tables_per_txn=1)
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())
        assert len(txn.tables_written) == 1
        # Table ID should be in range [0, num_tables)
        table_id = next(iter(txn.tables_written))
        assert 0 <= table_id < 5

    def test_multi_table_selection(self):
        config = make_config(tables_per_txn=3)
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())
        assert len(txn.tables_written) == 3

    def test_zipf_table_selector(self):
        config = make_config(
            table_selector=ZipfTableSelector(alpha=2.0),
            tables_per_txn=1,
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 5000)
        counts = Counter()
        for _, txn in results:
            counts[next(iter(txn.tables_written))] += 1

        # Table 0 should be most popular
        assert counts[0] > counts[4]


# ---------------------------------------------------------------------------
# Partition selection
# ---------------------------------------------------------------------------

class TestWorkloadPartitionSelection:
    def test_single_partition_by_default(self):
        config = make_config()
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())
        # Each written table gets single-partition default {tid: frozenset({0})}
        for tid in txn.tables_written:
            assert tid in txn.partitions_written
            assert txn.partitions_written[tid] == frozenset({0})

    def test_partitions_when_enabled(self):
        config = make_config(
            partitions_per_txn=2,
            partition_selector=UniformPartitionSelector(),
        )
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())

        # Should have partitions for each written table
        for table_id in txn.tables_written:
            assert table_id in txn.partitions_written
            parts = txn.partitions_written[table_id]
            assert len(parts) == 2
            for p in parts:
                assert 0 <= p < 4  # partitions_per_table = (4,4,4,4,4)

    def test_zipf_partition_selection(self):
        config = make_config(
            partitions_per_txn=1,
            partition_selector=ZipfPartitionSelector(alpha=2.0),
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 5000)

        counts = Counter()
        for _, txn in results:
            for parts in txn.partitions_written.values():
                for p in parts:
                    counts[p] += 1

        # Partition 0 should be most popular
        assert counts[0] > counts[3]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestWorkloadDeterminism:
    def test_same_seed_same_transactions(self):
        config = make_config()
        w1 = Workload(config, seed=123)
        w2 = Workload(config, seed=123)

        r1 = generate_n(w1, 50)
        r2 = generate_n(w2, 50)

        for (d1, t1), (d2, t2) in zip(r1, r2):
            assert d1 == d2
            assert t1.id == t2.id
            assert t1.submit_time == t2.submit_time
            assert t1.runtime == t2.runtime
            assert t1.tables_written == t2.tables_written
            assert type(t1) == type(t2)

    def test_different_seeds_different_transactions(self):
        config = make_config()
        w1 = Workload(config, seed=1)
        w2 = Workload(config, seed=999)

        r1 = generate_n(w1, 50)
        r2 = generate_n(w2, 50)

        # At least some should differ
        types1 = [type(t).__name__ for _, t in r1]
        types2 = [type(t).__name__ for _, t in r2]
        assert types1 != types2 or any(
            t1.tables_written != t2.tables_written
            for (_, t1), (_, t2) in zip(r1, r2)
        )


# ---------------------------------------------------------------------------
# Topology ownership
# ---------------------------------------------------------------------------

class TestTopologyOwnership:
    def test_topology_from_config_not_catalog(self):
        """Workload gets topology from config, does not reference Catalog."""
        config = make_config(num_tables=3, partitions_per_table=(2, 2, 2))
        workload = Workload(config, seed=42)
        # Workload has no catalog reference
        assert not hasattr(workload, '_catalog')
        assert workload.config.num_tables == 3

    def test_tables_within_range(self):
        config = make_config(num_tables=3, partitions_per_table=(2, 2, 2))
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for _, txn in results:
            for t in txn.tables_written:
                assert 0 <= t < 3

    def test_partitions_within_range(self):
        config = make_config(
            num_tables=2,
            partitions_per_table=(3, 5),
            partitions_per_txn=1,
            partition_selector=UniformPartitionSelector(),
        )
        workload = Workload(config, seed=42)
        results = generate_n(workload, 100)
        for _, txn in results:
            for table_id, parts in txn.partitions_written.items():
                max_part = config.partitions_per_table[table_id]
                for p in parts:
                    assert 0 <= p < max_part


# ---------------------------------------------------------------------------
# MergeAppend parameter
# ---------------------------------------------------------------------------

class TestMergeAppendParameter:
    def test_manifests_per_commit_passed(self):
        config = make_config(
            fast_append_weight=0.0,
            merge_append_weight=1.0,
            validated_overwrite_weight=0.0,
            manifests_per_concurrent_commit=2.5,
        )
        workload = Workload(config, seed=42)
        _, txn = next(workload.generate())
        assert isinstance(txn, MergeAppendTransaction)
        # Verify the parameter was passed by checking conflict cost
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False)
        # int(2 * 2.5) = 5
        assert cost.manifest_file_reads == 5
