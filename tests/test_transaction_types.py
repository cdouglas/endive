"""Unit tests for Transaction types per SPEC.md §3.

Tests:
- TransactionStatus enum values
- ConflictCost frozen dataclass with correct defaults
- TransactionResult frozen dataclass
- FastAppendTransaction: no real conflicts, cheap retry cost
- MergeAppendTransaction: no real conflicts, manifest re-merge cost scales
- ValidatedOverwriteTransaction: real conflicts abort, I/O convoy cost
- ML+ mode: ml_writes=0 for all types
"""

import pytest

from endive.transaction import (
    ConflictCost,
    ConflictDetector,
    FastAppendTransaction,
    MergeAppendTransaction,
    NO_OVERLAP,
    Transaction,
    TransactionResult,
    TransactionStatus,
    ValidatedOverwriteTransaction,
    WriteOverlap,
)
from endive.catalog import CatalogSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_txn(cls, txn_id=1, submit_time=0.0, runtime=100.0,
             tables_written=None, **kwargs):
    """Create a transaction with sensible defaults."""
    if tables_written is None:
        tables_written = frozenset({0})
    return cls(
        txn_id=txn_id,
        submit_time_ms=submit_time,
        runtime_ms=runtime,
        tables_written=tables_written,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# TransactionStatus
# ---------------------------------------------------------------------------

class TestTransactionStatus:
    def test_has_all_states(self):
        assert TransactionStatus.PENDING is not None
        assert TransactionStatus.EXECUTING is not None
        assert TransactionStatus.COMMITTING is not None
        assert TransactionStatus.COMMITTED is not None
        assert TransactionStatus.ABORTED is not None

    def test_distinct_values(self):
        statuses = [s for s in TransactionStatus]
        assert len(statuses) == 5
        assert len(set(s.value for s in statuses)) == 5


# ---------------------------------------------------------------------------
# ConflictCost
# ---------------------------------------------------------------------------

class TestConflictCost:
    def test_frozen(self):
        cost = ConflictCost()
        with pytest.raises(AttributeError):
            cost.metadata_reads = 5

    def test_defaults_are_zero(self):
        cost = ConflictCost()
        assert cost.metadata_reads == 0
        assert cost.manifest_list_reads == 0
        assert cost.manifest_list_writes == 0
        assert cost.historical_ml_reads == 0
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0

    def test_custom_values(self):
        cost = ConflictCost(
            metadata_reads=1,
            manifest_list_reads=2,
            manifest_list_writes=3,
            historical_ml_reads=4,
            manifest_file_reads=5,
            manifest_file_writes=6,
        )
        assert cost.metadata_reads == 1
        assert cost.manifest_list_reads == 2
        assert cost.manifest_list_writes == 3
        assert cost.historical_ml_reads == 4
        assert cost.manifest_file_reads == 5
        assert cost.manifest_file_writes == 6

    def test_equality(self):
        a = ConflictCost(metadata_reads=1, manifest_list_reads=1)
        b = ConflictCost(metadata_reads=1, manifest_list_reads=1)
        assert a == b

    def test_inequality(self):
        a = ConflictCost(metadata_reads=1)
        b = ConflictCost(metadata_reads=2)
        assert a != b


# ---------------------------------------------------------------------------
# TransactionResult
# ---------------------------------------------------------------------------

class TestTransactionResult:
    def test_frozen(self):
        result = TransactionResult(
            status=TransactionStatus.COMMITTED,
            txn_id=1,
            commit_time_ms=100.0,
            abort_time_ms=-1.0,
            abort_reason=None,
            total_retries=0,
            commit_latency_ms=5.0,
            total_latency_ms=105.0,
            operation_type="fast_append",
            runtime_ms=100.0,
            manifest_list_reads=0,
            manifest_list_writes=0,
            manifest_file_reads=0,
            manifest_file_writes=0,
        )
        with pytest.raises(AttributeError):
            result.status = TransactionStatus.ABORTED

    def test_committed_result(self):
        result = TransactionResult(
            status=TransactionStatus.COMMITTED,
            txn_id=42,
            commit_time_ms=200.0,
            abort_time_ms=-1.0,
            abort_reason=None,
            total_retries=2,
            commit_latency_ms=10.0,
            total_latency_ms=110.0,
            operation_type="fast_append",
            runtime_ms=100.0,
            manifest_list_reads=3,
            manifest_list_writes=1,
            manifest_file_reads=0,
            manifest_file_writes=0,
        )
        assert result.status == TransactionStatus.COMMITTED
        assert result.txn_id == 42
        assert result.commit_time_ms == 200.0
        assert result.abort_time_ms == -1.0
        assert result.abort_reason is None
        assert result.total_retries == 2

    def test_aborted_result(self):
        result = TransactionResult(
            status=TransactionStatus.ABORTED,
            txn_id=7,
            commit_time_ms=-1.0,
            abort_time_ms=150.0,
            abort_reason="validation_exception",
            total_retries=1,
            commit_latency_ms=8.0,
            total_latency_ms=108.0,
            operation_type="validated_overwrite",
            runtime_ms=100.0,
            manifest_list_reads=2,
            manifest_list_writes=0,
            manifest_file_reads=0,
            manifest_file_writes=0,
        )
        assert result.status == TransactionStatus.ABORTED
        assert result.commit_time_ms == -1.0
        assert result.abort_time_ms == 150.0
        assert result.abort_reason == "validation_exception"


# ---------------------------------------------------------------------------
# Transaction base class
# ---------------------------------------------------------------------------

class TestTransactionBase:
    def test_initial_status_is_pending(self):
        txn = make_txn(FastAppendTransaction)
        assert txn.status == TransactionStatus.PENDING

    def test_stores_attributes(self):
        txn = make_txn(
            FastAppendTransaction,
            txn_id=5,
            submit_time=100.0,
            runtime=200.0,
            tables_written=frozenset({0, 1}),
        )
        assert txn.id == 5
        assert txn.submit_time == 100.0
        assert txn.runtime == 200.0
        assert txn.tables_written == frozenset({0, 1})

    def test_default_partitions_written(self):
        txn = make_txn(FastAppendTransaction)
        assert txn.partitions_written == {}

    def test_custom_partitions_written(self):
        txn = make_txn(
            FastAppendTransaction,
            partitions_written={0: frozenset({1, 2})},
        )
        assert txn.partitions_written == {0: frozenset({1, 2})}


# ---------------------------------------------------------------------------
# FastAppendTransaction
# ---------------------------------------------------------------------------

class TestFastAppendTransaction:
    def test_cannot_have_real_conflict(self):
        txn = make_txn(FastAppendTransaction)
        assert txn.can_have_real_conflict() is False

    def test_should_not_abort_on_real_conflict(self):
        txn = make_txn(FastAppendTransaction)
        assert txn.should_abort_on_real_conflict() is False

    def test_conflict_cost_standard_mode(self):
        """FastAppend has no additional retry cost (per-attempt covers everything)."""
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost == ConflictCost()  # Empty — no additional retry cost

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=True)
        assert cost == ConflictCost()  # Still empty in ML+ mode

    def test_conflict_cost_independent_of_n_behind(self):
        """FastAppend retry cost is constant regardless of how far behind."""
        txn = make_txn(FastAppendTransaction)
        cost_1 = txn.get_conflict_cost(n_snapshots_behind=1, ml_append_mode=False)
        cost_10 = txn.get_conflict_cost(n_snapshots_behind=10, ml_append_mode=False)
        assert cost_1 == cost_10

    def test_no_historical_ml_reads(self):
        """FastAppend never reads historical manifest lists (no I/O convoy)."""
        txn = make_txn(FastAppendTransaction)
        for n in [0, 1, 5, 100]:
            cost = txn.get_conflict_cost(n_snapshots_behind=n, ml_append_mode=False)
            assert cost.historical_ml_reads == 0


# ---------------------------------------------------------------------------
# MergeAppendTransaction
# ---------------------------------------------------------------------------

class TestMergeAppendTransaction:
    def test_cannot_have_real_conflict(self):
        txn = make_txn(MergeAppendTransaction)
        assert txn.can_have_real_conflict() is False

    def test_should_not_abort_on_real_conflict(self):
        txn = make_txn(MergeAppendTransaction)
        assert txn.should_abort_on_real_conflict() is False

    def test_conflict_cost_standard_mode(self):
        """MergeAppend retry cost is only re-merge manifests (per-attempt is separate)."""
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False)
        assert cost.metadata_reads == 0    # Moved to per-attempt
        assert cost.manifest_list_reads == 0  # Moved to per-attempt
        assert cost.manifest_list_writes == 0  # Moved to per-attempt
        assert cost.historical_ml_reads == 0
        # Re-merge: int(2 * 1.5) = 3 manifests
        assert cost.manifest_file_reads == 3
        assert cost.manifest_file_writes == 3

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=True)
        assert cost.manifest_list_writes == 0  # ML+ doesn't affect retry cost

    def test_manifest_file_io_scales_with_n_behind(self):
        """MergeAppend manifest file I/O scales with snapshots behind."""
        txn = make_txn(MergeAppendTransaction, manifests_per_concurrent_commit=1.5)
        cost_1 = txn.get_conflict_cost(n_snapshots_behind=1, ml_append_mode=False)
        cost_4 = txn.get_conflict_cost(n_snapshots_behind=4, ml_append_mode=False)
        # n_behind=1: int(1 * 1.5) = 1 manifest
        assert cost_1.manifest_file_reads == 1
        assert cost_1.manifest_file_writes == 1
        # n_behind=4: int(4 * 1.5) = 6 manifests
        assert cost_4.manifest_file_reads == 6
        assert cost_4.manifest_file_writes == 6

    def test_manifest_per_commit_parameter(self):
        """Custom manifests_per_concurrent_commit parameter."""
        txn = make_txn(MergeAppendTransaction, manifests_per_concurrent_commit=2.0)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        # int(3 * 2.0) = 6
        assert cost.manifest_file_reads == 6
        assert cost.manifest_file_writes == 6

    def test_zero_manifests_when_zero_behind(self):
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=0, ml_append_mode=False)
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0

    def test_no_historical_ml_reads(self):
        """MergeAppend never reads historical manifest lists."""
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=10, ml_append_mode=False)
        assert cost.historical_ml_reads == 0

    def test_default_manifests_per_commit(self):
        """Default is 1.5 manifests per concurrent commit."""
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False)
        # int(2 * 1.5) = 3
        assert cost.manifest_file_reads == 3


# ---------------------------------------------------------------------------
# ValidatedOverwriteTransaction
# ---------------------------------------------------------------------------

class TestValidatedOverwriteTransaction:
    def test_can_have_real_conflict(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        assert txn.can_have_real_conflict() is True

    def test_should_abort_on_real_conflict(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        assert txn.should_abort_on_real_conflict() is True

    def test_conflict_cost_standard_mode(self):
        """ValidatedOverwrite retry cost is only I/O convoy (per-attempt is separate)."""
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost.metadata_reads == 0       # Moved to per-attempt
        assert cost.manifest_list_reads == 0  # Moved to per-attempt
        assert cost.manifest_list_writes == 0  # Moved to per-attempt
        assert cost.historical_ml_reads == 2  # I/O convoy: 3-1 (per-attempt reads current)
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=True)
        assert cost.manifest_list_writes == 0  # ML+ doesn't affect retry cost

    def test_historical_ml_reads_scale_with_n_behind(self):
        """I/O convoy: historical ML reads = n_snapshots_behind - 1 (per-attempt reads current)."""
        txn = make_txn(ValidatedOverwriteTransaction)
        for n in [1, 5, 10, 50]:
            cost = txn.get_conflict_cost(n_snapshots_behind=n, ml_append_mode=False)
            assert cost.historical_ml_reads == n - 1

    def test_zero_historical_reads_when_zero_behind(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=0, ml_append_mode=False)
        assert cost.historical_ml_reads == 0

    def test_no_manifest_file_io(self):
        """ValidatedOverwrite doesn't re-merge manifests (it aborts instead)."""
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=5, ml_append_mode=False)
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0


# ---------------------------------------------------------------------------
# ML+ mode across all types
# ---------------------------------------------------------------------------

class TestMLAppendMode:
    """ML+ (manifest list append) mode eliminates ML writes on conflict."""

    @pytest.mark.parametrize("cls", [
        FastAppendTransaction,
        MergeAppendTransaction,
        ValidatedOverwriteTransaction,
    ])
    def test_ml_append_mode_no_ml_writes(self, cls):
        """ML+ mode skips ML writes in per-attempt cost."""
        txn = make_txn(cls)
        per_attempt = txn.get_per_attempt_cost(ml_append_mode=True)
        assert per_attempt.manifest_list_writes == 0

    @pytest.mark.parametrize("cls", [
        FastAppendTransaction,
        MergeAppendTransaction,
        ValidatedOverwriteTransaction,
    ])
    def test_standard_mode_per_attempt_has_ml_writes(self, cls):
        """ML writes are in per-attempt cost (not retry cost) in standard mode."""
        txn = make_txn(cls)
        per_attempt = txn.get_per_attempt_cost(ml_append_mode=False)
        assert per_attempt.manifest_list_writes == 1


# ---------------------------------------------------------------------------
# ConflictDetector interface
# ---------------------------------------------------------------------------

class TestConflictDetectorInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ConflictDetector()

    def test_concrete_implementation(self):
        class MyDetector(ConflictDetector):
            def is_real_conflict(self, txn, current, start):
                return False

        detector = MyDetector()
        assert detector.is_real_conflict(None, None, None) is False


# ---------------------------------------------------------------------------
# has_write_overlap()
# ---------------------------------------------------------------------------

from endive.catalog import CatalogSnapshot, TableMetadata


def _make_snapshot(seq, table_versions, partition_versions_map):
    """Build a CatalogSnapshot with given table/partition versions.

    Args:
        seq: Catalog sequence number
        table_versions: list of table version ints
        partition_versions_map: dict {table_id: tuple of partition versions}
    """
    tables = []
    for tid, tv in enumerate(table_versions):
        pvs = partition_versions_map.get(tid, (0,))
        tables.append(TableMetadata(
            table_id=tid,
            version=tv,
            num_partitions=len(pvs),
            partition_versions=pvs,
        ))
    return CatalogSnapshot(seq=seq, tables=tuple(tables), timestamp_ms=0.0)


class TestHasWriteOverlap:
    """Tests for Transaction.has_write_overlap() per SPEC.md §3.3."""

    def test_cross_table_no_overlap(self):
        """T writes table 1, intervening commit was to table 0 → False."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({1}),
            partitions_written={1: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0,), 1: (0,)})
        new = _make_snapshot(1, [1, 0], {0: (1,), 1: (0,)})  # table 0 modified
        assert txn.has_write_overlap(old, new) is False

    def test_same_table_disjoint_partitions(self):
        """T writes table 0 p1, intervening commit to table 0 p0 → False."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({1})},
        )
        old = _make_snapshot(0, [0], {0: (0, 0)})
        new = _make_snapshot(1, [1], {0: (1, 0)})  # p0 changed, p1 unchanged
        assert txn.has_write_overlap(old, new) is False

    def test_same_table_overlapping_partitions(self):
        """Both write table 0 p0 → True."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({0})},
        )
        old = _make_snapshot(0, [0], {0: (0,)})
        new = _make_snapshot(1, [1], {0: (1,)})  # p0 changed
        assert txn.has_write_overlap(old, new) is True

    def test_multi_table_partial_overlap_disjoint(self):
        """T writes {0, 1}, table 0 modified but disjoint partitions → False."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0, 1}),
            partitions_written={0: frozenset({1}), 1: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0, 0), 1: (0,)})
        new = _make_snapshot(1, [1, 0], {0: (1, 0), 1: (0,)})  # table 0 p0 changed
        assert txn.has_write_overlap(old, new) is False

    def test_multi_table_partial_overlap_overlapping(self):
        """T writes {0, 1}, table 0 modified with overlapping partition → True."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0, 1}),
            partitions_written={0: frozenset({0}), 1: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0,), 1: (0,)})
        new = _make_snapshot(1, [1, 0], {0: (1,), 1: (0,)})  # table 0 p0 changed
        assert txn.has_write_overlap(old, new) is True

    def test_no_table_modified(self):
        """Intervening commit to unrelated table → False."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0,), 1: (0,)})
        new = _make_snapshot(1, [0, 1], {0: (0,), 1: (1,)})  # only table 1 changed
        assert txn.has_write_overlap(old, new) is False

    def test_multiple_snapshots_behind(self):
        """seq advances by 3, any overlap → True."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({0})},
        )
        old = _make_snapshot(0, [0], {0: (0,)})
        new = _make_snapshot(3, [3], {0: (3,)})  # 3 commits to same table/partition
        assert txn.has_write_overlap(old, new) is True


# ---------------------------------------------------------------------------
# WriteOverlap dataclass
# ---------------------------------------------------------------------------

class TestWriteOverlap:
    def test_no_overlap_sentinel(self):
        assert NO_OVERLAP.has_overlap is False
        assert NO_OVERLAP.n_tables == 0
        assert NO_OVERLAP.n_partitions == 0

    def test_single_table_single_partition(self):
        overlap = WriteOverlap(overlapping={0: frozenset({0})})
        assert overlap.has_overlap is True
        assert overlap.n_tables == 1
        assert overlap.n_partitions == 1

    def test_multi_table(self):
        overlap = WriteOverlap(overlapping={0: frozenset({0}), 2: frozenset({1, 3})})
        assert overlap.has_overlap is True
        assert overlap.n_tables == 2
        assert overlap.n_partitions == 3

    def test_frozen(self):
        with pytest.raises(AttributeError):
            NO_OVERLAP.overlapping = {0: frozenset({0})}

    def test_equality(self):
        a = WriteOverlap(overlapping={0: frozenset({0})})
        b = WriteOverlap(overlapping={0: frozenset({0})})
        assert a == b

    def test_empty_overlapping_is_no_overlap(self):
        empty = WriteOverlap(overlapping={})
        assert empty.has_overlap is False
        assert empty == NO_OVERLAP


# ---------------------------------------------------------------------------
# compute_write_overlap()
# ---------------------------------------------------------------------------

class TestComputeWriteOverlap:
    def test_cross_table_returns_no_overlap(self):
        """T writes table 1, intervening commit to table 0 → NO_OVERLAP."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({1}),
            partitions_written={1: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0,), 1: (0,)})
        new = _make_snapshot(1, [1, 0], {0: (1,), 1: (0,)})
        result = txn.compute_write_overlap(old, new)
        assert result is NO_OVERLAP

    def test_same_table_overlapping_returns_correct_dict(self):
        """Both write table 0 p0 → overlap = {0: {0}}."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({0})},
        )
        old = _make_snapshot(0, [0], {0: (0,)})
        new = _make_snapshot(1, [1], {0: (1,)})
        result = txn.compute_write_overlap(old, new)
        assert result.has_overlap is True
        assert result.overlapping == {0: frozenset({0})}
        assert result.n_partitions == 1

    def test_multi_table_partial_overlap(self):
        """T writes {A.0, B.0, B.2}. Intervening modifies {A.1, B.0}.
        Overlap = {B: {0}}, n_partitions=1."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0, 1}),
            partitions_written={0: frozenset({0}), 1: frozenset({0, 2})},
        )
        # Table 0 (A): 2 partitions, p1 modified (txn writes p0 → no overlap)
        # Table 1 (B): 3 partitions, p0 modified (txn writes p0,p2 → overlap on p0)
        old = _make_snapshot(0, [0, 0], {0: (0, 0), 1: (0, 0, 0)})
        new = _make_snapshot(2, [1, 1], {0: (0, 1), 1: (1, 0, 0)})
        result = txn.compute_write_overlap(old, new)
        assert result.has_overlap is True
        assert result.overlapping == {1: frozenset({0})}
        assert result.n_tables == 1
        assert result.n_partitions == 1

    def test_disjoint_partitions_returns_no_overlap(self):
        """Same table but disjoint partitions → NO_OVERLAP."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({1})},
        )
        old = _make_snapshot(0, [0], {0: (0, 0)})
        new = _make_snapshot(1, [1], {0: (1, 0)})
        result = txn.compute_write_overlap(old, new)
        assert result is NO_OVERLAP

    def test_backward_compat_with_has_write_overlap(self):
        """compute_write_overlap().has_overlap == has_write_overlap()."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0, 1}),
            partitions_written={0: frozenset({0}), 1: frozenset({0})},
        )
        old = _make_snapshot(0, [0, 0], {0: (0,), 1: (0,)})
        new = _make_snapshot(1, [1, 0], {0: (1,), 1: (0,)})
        assert txn.compute_write_overlap(old, new).has_overlap == txn.has_write_overlap(old, new)

    def test_multi_partition_overlap_counts(self):
        """T writes {A.0, A.1, A.2}. All modified → n_partitions=3."""
        txn = make_txn(
            FastAppendTransaction,
            tables_written=frozenset({0}),
            partitions_written={0: frozenset({0, 1, 2})},
        )
        old = _make_snapshot(0, [0], {0: (0, 0, 0)})
        new = _make_snapshot(1, [1], {0: (1, 1, 1)})
        result = txn.compute_write_overlap(old, new)
        assert result.n_partitions == 3
        assert result.overlapping == {0: frozenset({0, 1, 2})}


# ---------------------------------------------------------------------------
# Scaled per-attempt cost
# ---------------------------------------------------------------------------

class TestScaledPerAttemptCost:
    def test_default_n_partitions_unchanged(self):
        """Default n_partitions=1 → same as before."""
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_per_attempt_cost(ml_append_mode=False)
        assert cost.manifest_list_reads == 1
        assert cost.manifest_file_writes == 1
        assert cost.manifest_list_writes == 1

    def test_n_partitions_scales_all_fields(self):
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_per_attempt_cost(ml_append_mode=False, n_partitions=3)
        assert cost.manifest_list_reads == 3
        assert cost.manifest_file_writes == 3
        assert cost.manifest_list_writes == 3

    def test_ml_append_mode_with_n_partitions(self):
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_per_attempt_cost(ml_append_mode=True, n_partitions=5)
        assert cost.manifest_list_reads == 5
        assert cost.manifest_file_writes == 5
        assert cost.manifest_list_writes == 0  # ML+ suppresses ML writes


# ---------------------------------------------------------------------------
# Scaled conflict cost
# ---------------------------------------------------------------------------

class TestScaledConflictCost:
    def test_fast_append_ignores_n_partitions(self):
        """FastAppend always returns empty ConflictCost regardless of n_partitions."""
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False, n_partitions=10)
        assert cost == ConflictCost()

    def test_merge_append_scales_by_n_partitions(self):
        """MergeAppend: n_manifests = int(n_behind * manifests_per) * n_partitions."""
        txn = make_txn(MergeAppendTransaction, manifests_per_concurrent_commit=2.0)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False, n_partitions=3)
        # int(2 * 2.0) * 3 = 12
        assert cost.manifest_file_reads == 12
        assert cost.manifest_file_writes == 12

    def test_validated_overwrite_scales_by_n_partitions(self):
        """ValidatedOverwrite: historical_ml_reads = (n_behind - 1) * n_partitions."""
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False, n_partitions=5)
        assert cost.historical_ml_reads == 10  # (3-1) * 5

    def test_default_n_partitions_unchanged_merge_append(self):
        txn = make_txn(MergeAppendTransaction, manifests_per_concurrent_commit=2.0)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False)
        # int(2 * 2.0) * 1 = 4
        assert cost.manifest_file_reads == 4
        assert cost.manifest_file_writes == 4

    def test_default_n_partitions_unchanged_validated_overwrite(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost.historical_ml_reads == 2  # 3-1, per-attempt reads current
