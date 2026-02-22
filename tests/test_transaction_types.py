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
    Transaction,
    TransactionResult,
    TransactionStatus,
    ValidatedOverwriteTransaction,
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
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost.metadata_reads == 1
        assert cost.manifest_list_reads == 1
        assert cost.manifest_list_writes == 1
        assert cost.historical_ml_reads == 0
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(FastAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=True)
        assert cost.manifest_list_writes == 0

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
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=False)
        assert cost.metadata_reads == 1
        assert cost.manifest_list_reads == 1
        assert cost.manifest_list_writes == 1
        assert cost.historical_ml_reads == 0

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(MergeAppendTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=2, ml_append_mode=True)
        assert cost.manifest_list_writes == 0

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
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost.metadata_reads == 1
        assert cost.manifest_list_reads == 1
        assert cost.manifest_list_writes == 1
        assert cost.historical_ml_reads == 3  # I/O convoy
        assert cost.manifest_file_reads == 0
        assert cost.manifest_file_writes == 0

    def test_conflict_cost_ml_append_mode(self):
        txn = make_txn(ValidatedOverwriteTransaction)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=True)
        assert cost.manifest_list_writes == 0

    def test_historical_ml_reads_scale_with_n_behind(self):
        """I/O convoy: historical ML reads = n_snapshots_behind."""
        txn = make_txn(ValidatedOverwriteTransaction)
        for n in [1, 5, 10, 50]:
            cost = txn.get_conflict_cost(n_snapshots_behind=n, ml_append_mode=False)
            assert cost.historical_ml_reads == n

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
        txn = make_txn(cls)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=True)
        assert cost.manifest_list_writes == 0

    @pytest.mark.parametrize("cls", [
        FastAppendTransaction,
        MergeAppendTransaction,
        ValidatedOverwriteTransaction,
    ])
    def test_standard_mode_has_ml_writes(self, cls):
        txn = make_txn(cls)
        cost = txn.get_conflict_cost(n_snapshots_behind=3, ml_append_mode=False)
        assert cost.manifest_list_writes == 1


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
