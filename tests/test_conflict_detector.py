"""Tests for ConflictDetector implementations per SPEC.md §5.

Tests:
- ProbabilisticConflictDetector:
  - prob=0: never real conflict
  - prob=1: always real for ValidatedOverwrite
  - Respects can_have_real_conflict() (FastAppend immune)
  - Deterministic with seeded RNG
  - Validates probability range
- PartitionOverlapConflictDetector:
  - Detects same-partition modification (real conflict)
  - Different partitions → false conflict
  - No partitions written → false conflict
  - Respects can_have_real_conflict()
  - Multiple tables
"""

import pytest
import numpy as np

from endive.catalog import CatalogSnapshot, TableMetadata
from endive.conflict_detector import (
    PartitionOverlapConflictDetector,
    ProbabilisticConflictDetector,
)
from endive.transaction import (
    ConflictDetector,
    FastAppendTransaction,
    MergeAppendTransaction,
    ValidatedOverwriteTransaction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_snapshot(seq=0, table_versions=None, partition_versions=None):
    """Create a CatalogSnapshot for testing.

    Args:
        seq: Catalog sequence number
        table_versions: List of (table_id, version, num_partitions) or just version ints
        partition_versions: Dict of table_id -> tuple of partition versions
    """
    if table_versions is None:
        table_versions = [0]

    tables = []
    for i, tv in enumerate(table_versions):
        if isinstance(tv, int):
            num_parts = 4  # default
            pv = partition_versions.get(i, tuple([0] * num_parts)) if partition_versions else tuple([0] * num_parts)
            tables.append(TableMetadata(
                table_id=i,
                version=tv,
                num_partitions=len(pv),
                partition_versions=pv,
            ))
        else:
            tables.append(tv)

    return CatalogSnapshot(
        seq=seq,
        tables=tuple(tables),
        timestamp_ms=0.0,
    )


def make_fast_append(**kwargs):
    defaults = dict(
        txn_id=1,
        submit_time_ms=0.0,
        runtime_ms=100.0,
        tables_written=frozenset({0}),
    )
    defaults.update(kwargs)
    return FastAppendTransaction(**defaults)


def make_merge_append(**kwargs):
    defaults = dict(
        txn_id=1,
        submit_time_ms=0.0,
        runtime_ms=100.0,
        tables_written=frozenset({0}),
    )
    defaults.update(kwargs)
    return MergeAppendTransaction(**defaults)


def make_validated_overwrite(**kwargs):
    defaults = dict(
        txn_id=1,
        submit_time_ms=0.0,
        runtime_ms=100.0,
        tables_written=frozenset({0}),
    )
    defaults.update(kwargs)
    return ValidatedOverwriteTransaction(**defaults)


# ---------------------------------------------------------------------------
# ProbabilisticConflictDetector
# ---------------------------------------------------------------------------

class TestProbabilisticConflictDetector:
    def test_prob_zero_never_real(self):
        """prob=0 → never real conflict for any operation type."""
        detector = ProbabilisticConflictDetector(0.0, rng=np.random.RandomState(42))
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        txn = make_validated_overwrite()
        for _ in range(100):
            assert detector.is_real_conflict(txn, current, start) is False

    def test_prob_one_always_real_for_validated(self):
        """prob=1 → always real for ValidatedOverwrite."""
        detector = ProbabilisticConflictDetector(1.0, rng=np.random.RandomState(42))
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        txn = make_validated_overwrite()
        for _ in range(100):
            assert detector.is_real_conflict(txn, current, start) is True

    def test_fast_append_immune(self):
        """FastAppend never returns real conflict regardless of probability."""
        detector = ProbabilisticConflictDetector(1.0, rng=np.random.RandomState(42))
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        txn = make_fast_append()
        assert detector.is_real_conflict(txn, current, start) is False

    def test_merge_append_immune(self):
        """MergeAppend never returns real conflict regardless of probability."""
        detector = ProbabilisticConflictDetector(1.0, rng=np.random.RandomState(42))
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        txn = make_merge_append()
        assert detector.is_real_conflict(txn, current, start) is False

    def test_deterministic_with_seed(self):
        """Same seed produces identical sequence of decisions."""
        results_a = []
        results_b = []
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        for seed_results, seed in [(results_a, 123), (results_b, 123)]:
            detector = ProbabilisticConflictDetector(0.5, rng=np.random.RandomState(seed))
            txn = make_validated_overwrite()
            for _ in range(50):
                seed_results.append(detector.is_real_conflict(txn, current, start))

        assert results_a == results_b

    def test_different_seeds_different_results(self):
        """Different seeds produce different sequences."""
        results_a = []
        results_b = []
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)

        for seed_results, seed in [(results_a, 1), (results_b, 999)]:
            detector = ProbabilisticConflictDetector(0.5, rng=np.random.RandomState(seed))
            txn = make_validated_overwrite()
            for _ in range(50):
                seed_results.append(detector.is_real_conflict(txn, current, start))

        assert results_a != results_b

    def test_probability_approximately_correct(self):
        """Over many trials, fraction of real conflicts ≈ probability."""
        detector = ProbabilisticConflictDetector(0.3, rng=np.random.RandomState(42))
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)
        txn = make_validated_overwrite()

        n_trials = 10000
        n_real = sum(
            detector.is_real_conflict(txn, current, start)
            for _ in range(n_trials)
        )
        observed = n_real / n_trials
        assert abs(observed - 0.3) < 0.03  # Within 3% of expected

    def test_validates_probability_range(self):
        with pytest.raises(ValueError):
            ProbabilisticConflictDetector(-0.1)
        with pytest.raises(ValueError):
            ProbabilisticConflictDetector(1.1)

    def test_probability_bounds_accepted(self):
        ProbabilisticConflictDetector(0.0)
        ProbabilisticConflictDetector(1.0)

    def test_probability_property(self):
        detector = ProbabilisticConflictDetector(0.42)
        assert detector.probability == 0.42

    def test_default_rng_created(self):
        """If no RNG provided, a default is created."""
        detector = ProbabilisticConflictDetector(0.5)
        start = make_snapshot(seq=0)
        current = make_snapshot(seq=1)
        txn = make_validated_overwrite()
        # Should not raise
        detector.is_real_conflict(txn, current, start)


# ---------------------------------------------------------------------------
# PartitionOverlapConflictDetector
# ---------------------------------------------------------------------------

class TestPartitionOverlapConflictDetector:
    def test_same_partition_modified_is_real_conflict(self):
        """If txn writes partition 2 and it changed → real conflict."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            partitions_written={0: frozenset({2})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (0, 0, 1, 0)})  # Part 2 changed

        assert detector.is_real_conflict(txn, current, start) is True

    def test_different_partition_modified_is_false_conflict(self):
        """If txn writes partition 2 but partition 3 changed → false conflict."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            partitions_written={0: frozenset({2})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (0, 0, 0, 1)})  # Part 3 changed

        assert detector.is_real_conflict(txn, current, start) is False

    def test_no_partitions_written_is_false_conflict(self):
        """If txn writes no partitions → no conflict possible."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            partitions_written={},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (1, 1, 1, 1)})  # All changed

        assert detector.is_real_conflict(txn, current, start) is False

    def test_no_partition_change_is_false_conflict(self):
        """If written partition hasn't changed → false conflict."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            partitions_written={0: frozenset({0, 1})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (5, 3, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (5, 3, 0, 0)})  # No change

        assert detector.is_real_conflict(txn, current, start) is False

    def test_fast_append_immune(self):
        """FastAppend never returns real even with partition overlap."""
        detector = PartitionOverlapConflictDetector()
        txn = make_fast_append(
            partitions_written={0: frozenset({0})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (1, 0, 0, 0)})

        assert detector.is_real_conflict(txn, current, start) is False

    def test_merge_append_immune(self):
        """MergeAppend never returns real even with partition overlap."""
        detector = PartitionOverlapConflictDetector()
        txn = make_merge_append(
            partitions_written={0: frozenset({0})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (1, 0, 0, 0)})

        assert detector.is_real_conflict(txn, current, start) is False

    def test_multiple_tables(self):
        """Conflict detection across multiple tables."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            tables_written=frozenset({0, 1}),
            partitions_written={
                0: frozenset({0}),
                1: frozenset({2}),
            },
        )
        start = make_snapshot(
            seq=0,
            table_versions=[0, 0],
            partition_versions={
                0: (0, 0, 0, 0),
                1: (0, 0, 0, 0),
            },
        )
        # Only table 1 partition 2 changed → real conflict
        current = make_snapshot(
            seq=1,
            table_versions=[0, 0],
            partition_versions={
                0: (0, 0, 0, 0),
                1: (0, 0, 1, 0),
            },
        )
        assert detector.is_real_conflict(txn, current, start) is True

    def test_multiple_tables_no_overlap(self):
        """Changes in non-written partitions across tables."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            tables_written=frozenset({0, 1}),
            partitions_written={
                0: frozenset({0}),
                1: frozenset({2}),
            },
        )
        start = make_snapshot(
            seq=0,
            table_versions=[0, 0],
            partition_versions={
                0: (0, 0, 0, 0),
                1: (0, 0, 0, 0),
            },
        )
        # Table 0 part 3 and table 1 part 0 changed (not written partitions)
        current = make_snapshot(
            seq=1,
            table_versions=[0, 0],
            partition_versions={
                0: (0, 0, 0, 1),
                1: (1, 0, 0, 0),
            },
        )
        assert detector.is_real_conflict(txn, current, start) is False

    def test_multiple_written_partitions_one_overlaps(self):
        """Txn writes partitions {0, 2}, only partition 2 changed."""
        detector = PartitionOverlapConflictDetector()
        txn = make_validated_overwrite(
            partitions_written={0: frozenset({0, 2})},
        )
        start = make_snapshot(seq=0, partition_versions={0: (0, 0, 0, 0)})
        current = make_snapshot(seq=1, partition_versions={0: (0, 0, 1, 0)})

        assert detector.is_real_conflict(txn, current, start) is True


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestConflictDetectorInterface:
    def test_probabilistic_is_conflict_detector(self):
        detector = ProbabilisticConflictDetector(0.5)
        assert isinstance(detector, ConflictDetector)

    def test_partition_overlap_is_conflict_detector(self):
        detector = PartitionOverlapConflictDetector()
        assert isinstance(detector, ConflictDetector)
