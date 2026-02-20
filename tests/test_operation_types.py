"""Tests for operation types and conflict resolution behavior.

Verifies that:
1. FastAppend has zero historical ML reads (no I/O convoy)
2. ValidatedOverwrite has O(N) historical ML reads (I/O convoy)
3. Real conflicts on ValidatedOverwrite abort with ValidationException
4. MergeAppend has manifest file I/O proportional to n_behind
5. Statistics track validation_exceptions correctly
6. Operation type sampling works correctly
"""

import os
import tempfile
import pytest
import numpy as np

from endive.operation import (
    OperationType,
    ConflictCost,
    FastAppendBehavior,
    MergeAppendBehavior,
    ValidatedOverwriteBehavior,
    get_behavior,
    FAST_APPEND_BEHAVIOR,
    VALIDATED_OVERWRITE_BEHAVIOR,
)
from endive.conflict import resolve_conflict, ConflictResult
from endive.transaction import Txn
from endive.snapshot import CatalogSnapshot
from endive.capstats import Stats


class TestOperationBehaviors:
    """Test operation behavior cost models."""

    def test_fast_append_no_historical_ml_reads(self):
        """FastAppend should have zero historical ML reads (no I/O convoy)."""
        behavior = FAST_APPEND_BEHAVIOR

        # Even when 100 snapshots behind, no historical ML reads
        cost = behavior.get_false_conflict_cost(n_behind=100, ml_append_mode=False)

        assert cost.historical_ml_reads == 0, "FastAppend should have no historical ML reads"
        assert cost.ml_reads == 1, "FastAppend should read current ML"
        assert cost.ml_writes == 1, "FastAppend should write ML in rewrite mode"
        assert cost.manifest_file_reads == 0, "FastAppend should not read manifest files"
        assert cost.manifest_file_writes == 0, "FastAppend should not write manifest files"

    def test_fast_append_ml_append_mode(self):
        """FastAppend in ML+ mode should skip ML writes."""
        behavior = FAST_APPEND_BEHAVIOR

        cost = behavior.get_false_conflict_cost(n_behind=10, ml_append_mode=True)

        assert cost.ml_writes == 0, "FastAppend should skip ML writes in ML+ mode"
        assert cost.historical_ml_reads == 0, "Still no historical ML reads"

    def test_fast_append_cannot_have_real_conflict(self):
        """FastAppend cannot have real conflicts (appends are additive)."""
        behavior = FAST_APPEND_BEHAVIOR

        assert not behavior.can_have_real_conflict(), "FastAppend cannot have real conflicts"

    def test_validated_overwrite_io_convoy(self):
        """ValidatedOverwrite should have O(N) historical ML reads."""
        behavior = VALIDATED_OVERWRITE_BEHAVIOR

        # 10 snapshots behind
        cost = behavior.get_false_conflict_cost(n_behind=10, ml_append_mode=False)
        assert cost.historical_ml_reads == 10, "Should read 10 historical MLs"

        # 100 snapshots behind - the I/O convoy
        cost = behavior.get_false_conflict_cost(n_behind=100, ml_append_mode=False)
        assert cost.historical_ml_reads == 100, "Should read 100 historical MLs"

        # Basic ML operations still happen
        assert cost.ml_reads == 1, "Should read current ML"
        assert cost.ml_writes == 1, "Should write ML in rewrite mode"

    def test_validated_overwrite_can_have_real_conflict(self):
        """ValidatedOverwrite can detect real conflicts."""
        behavior = VALIDATED_OVERWRITE_BEHAVIOR

        assert behavior.can_have_real_conflict(), "ValidatedOverwrite can have real conflicts"

    def test_merge_append_manifest_file_scaling(self):
        """MergeAppend should scale manifest file I/O with n_behind."""
        behavior = MergeAppendBehavior(manifests_per_commit=2.0)

        # 5 commits behind * 2 manifests per commit = 10 manifest file ops
        cost = behavior.get_false_conflict_cost(n_behind=5, ml_append_mode=False)

        assert cost.manifest_file_reads == 10, "Should read 10 manifest files"
        assert cost.manifest_file_writes == 10, "Should write 10 manifest files"
        assert cost.historical_ml_reads == 0, "No I/O convoy for MergeAppend"

    def test_merge_append_cannot_have_real_conflict(self):
        """MergeAppend has no validation, so no real conflicts."""
        behavior = MergeAppendBehavior()

        assert not behavior.can_have_real_conflict(), "MergeAppend cannot have real conflicts"

    def test_get_behavior_factory(self):
        """get_behavior() should return correct implementation."""
        assert isinstance(get_behavior(OperationType.FAST_APPEND), FastAppendBehavior)
        assert isinstance(get_behavior(OperationType.MERGE_APPEND), MergeAppendBehavior)
        assert isinstance(get_behavior(OperationType.VALIDATED_OVERWRITE), ValidatedOverwriteBehavior)


class TestTransactionOperationType:
    """Test Txn.operation_type and get_behavior() method."""

    def test_txn_default_operation_type(self):
        """Transaction without operation_type should default to FastAppend behavior."""
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})

        assert txn.operation_type is None, "Default operation_type should be None"

        behavior = txn.get_behavior()
        assert isinstance(behavior, FastAppendBehavior), "Default behavior should be FastAppend"

    def test_txn_with_operation_type(self):
        """Transaction with operation_type should return correct behavior."""
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})
        txn.operation_type = OperationType.VALIDATED_OVERWRITE

        behavior = txn.get_behavior()
        assert isinstance(behavior, ValidatedOverwriteBehavior)

    def test_txn_abort_reason_field(self):
        """Transaction should have abort_reason field."""
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})

        assert txn.abort_reason is None, "Default abort_reason should be None"

        txn.abort_reason = "validation_exception"
        assert txn.abort_reason == "validation_exception"


class TestConflictResolution:
    """Test conflict resolution with operation types."""

    def _create_test_snapshot(self, seq: int, n_tables: int = 5) -> CatalogSnapshot:
        """Create a test snapshot."""
        return CatalogSnapshot(
            seq=seq,
            tbl=tuple([seq] * n_tables),
            partition_seq=None,
            ml_offset=tuple([1000] * n_tables),
            partition_ml_offset=None,
            timestamp=seq * 100,
        )

    def _create_test_txn(self, seq: int, op_type: OperationType = None) -> Txn:
        """Create a test transaction."""
        txn = Txn(1, 0, 100, seq, {0: seq}, {0: seq + 1})
        txn.v_dirty = {0: seq}
        txn.operation_type = op_type
        return txn

    def test_fast_append_always_retries(self):
        """FastAppend should always retry, never abort on real conflict."""
        import simpy
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                env = simpy.Environment()
                txn = self._create_test_txn(seq=5, op_type=OperationType.FAST_APPEND)
                snapshot = self._create_test_snapshot(seq=10)  # 5 behind

                def run():
                    result = yield from resolve_conflict(
                        env, txn, snapshot,
                        real_conflict_probability=1.0,  # 100% real conflict
                        manifest_list_mode="rewrite",
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run())
                env.run()

                result = process.value
                assert result.should_retry is True, "FastAppend should always retry"
                assert result.abort_reason is None, "No abort reason for retry"

            finally:
                os.unlink(config_path)

    def test_validated_overwrite_aborts_on_real_conflict(self):
        """ValidatedOverwrite should abort on real conflict with ValidationException."""
        import simpy
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                env = simpy.Environment()
                txn = self._create_test_txn(seq=5, op_type=OperationType.VALIDATED_OVERWRITE)
                snapshot = self._create_test_snapshot(seq=10)  # 5 behind

                # Set seed for deterministic "real conflict" result
                np.random.seed(0)  # With this seed, random() < 1.0 is always True

                def run():
                    result = yield from resolve_conflict(
                        env, txn, snapshot,
                        real_conflict_probability=1.0,  # 100% real conflict
                        manifest_list_mode="rewrite",
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run())
                env.run()

                result = process.value
                assert result.should_retry is False, "ValidatedOverwrite should abort on real conflict"
                assert result.abort_reason == "validation_exception", "Abort reason should be validation_exception"

                # Check stats
                assert endive.main.STATS.real_conflicts == 1, "Should count real conflict"
                assert endive.main.STATS.validation_exceptions == 1, "Should count validation exception"

            finally:
                os.unlink(config_path)

    def test_validated_overwrite_retries_on_false_conflict(self):
        """ValidatedOverwrite should retry on false conflict (no partition overlap)."""
        import simpy
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                env = simpy.Environment()
                txn = self._create_test_txn(seq=5, op_type=OperationType.VALIDATED_OVERWRITE)
                snapshot = self._create_test_snapshot(seq=10)  # 5 behind

                def run():
                    result = yield from resolve_conflict(
                        env, txn, snapshot,
                        real_conflict_probability=0.0,  # 0% real conflict (all false)
                        manifest_list_mode="rewrite",
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run())
                env.run()

                result = process.value
                assert result.should_retry is True, "Should retry on false conflict"
                assert result.abort_reason is None

                # Check stats - should have false conflict with historical ML reads
                assert endive.main.STATS.false_conflicts == 1, "Should count false conflict"
                # Historical ML reads = 5 (n_behind) + 1 current = 6 total
                # Note: The exact count depends on implementation details

            finally:
                os.unlink(config_path)


class TestStatsTracking:
    """Test that statistics track operation types and aborts correctly."""

    def test_stats_has_validation_exceptions_counter(self):
        """Stats should have validation_exceptions counter."""
        stats = Stats()

        assert hasattr(stats, 'validation_exceptions')
        assert stats.validation_exceptions == 0

    def test_abort_records_reason(self):
        """Stats.abort() should record abort_reason."""
        stats = Stats()
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})
        txn.t_abort = 200
        txn.abort_reason = "validation_exception"

        stats.abort(txn)

        assert len(stats.transactions) == 1
        record = stats.transactions[0]
        assert record['status'] == 'aborted'
        assert record['abort_reason'] == 'validation_exception'

    def test_abort_default_reason(self):
        """Stats.abort() should default to max_retries reason."""
        stats = Stats()
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})
        txn.t_abort = 200
        # No abort_reason set

        stats.abort(txn)

        record = stats.transactions[0]
        assert record['abort_reason'] == 'max_retries'

    def test_commit_records_operation_type(self):
        """Stats.commit() should record operation_type."""
        stats = Stats()
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})
        txn.t_commit = 200
        txn.operation_type = OperationType.VALIDATED_OVERWRITE

        stats.commit(txn)

        record = stats.transactions[0]
        assert record['operation_type'] == 'validated_overwrite'

    def test_commit_null_operation_type(self):
        """Stats.commit() should handle None operation_type."""
        stats = Stats()
        txn = Txn(1, 0, 100, 0, {0: 0}, {0: 1})
        txn.t_commit = 200
        # operation_type defaults to None

        stats.commit(txn)

        record = stats.transactions[0]
        assert record['operation_type'] is None


class TestOperationTypeSampling:
    """Test operation type sampling from configured weights."""

    def test_sample_operation_type_all_fast_append(self):
        """With 100% fast_append weight, should return None (legacy behavior)."""
        import endive.main
        from endive.main import _sample_operation_type

        endive.main.OPERATION_TYPE_WEIGHTS = {
            "fast_append": 1.0,
            "merge_append": 0.0,
            "validated_overwrite": 0.0,
        }

        # 100% fast_append returns None for backward compatibility
        # This enables legacy conflict resolution behavior
        for _ in range(100):
            op_type = _sample_operation_type()
            assert op_type is None, "100% fast_append should return None for legacy behavior"

    def test_sample_operation_type_explicit_fast_append(self):
        """With explicit mixed weights including fast_append, should sample FastAppend."""
        import endive.main
        from endive.main import _sample_operation_type

        # Mixed weights (not 100% fast_append) return actual OperationType values
        endive.main.OPERATION_TYPE_WEIGHTS = {
            "fast_append": 0.999,  # Not exactly 1.0
            "merge_append": 0.001,
            "validated_overwrite": 0.0,
        }

        np.random.seed(42)
        sampled_fast_append = False
        for _ in range(100):
            op_type = _sample_operation_type()
            assert op_type in [OperationType.FAST_APPEND, OperationType.MERGE_APPEND]
            if op_type == OperationType.FAST_APPEND:
                sampled_fast_append = True

        assert sampled_fast_append, "Should have sampled at least one FastAppend"

    def test_sample_operation_type_all_validated(self):
        """With 100% validated_overwrite weight, should always sample ValidatedOverwrite."""
        import endive.main
        from endive.main import _sample_operation_type

        endive.main.OPERATION_TYPE_WEIGHTS = {
            "fast_append": 0.0,
            "merge_append": 0.0,
            "validated_overwrite": 1.0,
        }

        for _ in range(100):
            op_type = _sample_operation_type()
            assert op_type == OperationType.VALIDATED_OVERWRITE

    def test_sample_operation_type_mixed(self):
        """With mixed weights, should sample proportionally."""
        import endive.main
        from endive.main import _sample_operation_type

        endive.main.OPERATION_TYPE_WEIGHTS = {
            "fast_append": 0.5,
            "merge_append": 0.3,
            "validated_overwrite": 0.2,
        }

        np.random.seed(42)
        counts = {
            OperationType.FAST_APPEND: 0,
            OperationType.MERGE_APPEND: 0,
            OperationType.VALIDATED_OVERWRITE: 0,
        }

        n_samples = 1000
        for _ in range(n_samples):
            op_type = _sample_operation_type()
            counts[op_type] += 1

        # Check proportions are approximately correct (within 10%)
        assert 0.4 <= counts[OperationType.FAST_APPEND] / n_samples <= 0.6
        assert 0.2 <= counts[OperationType.MERGE_APPEND] / n_samples <= 0.4
        assert 0.1 <= counts[OperationType.VALIDATED_OVERWRITE] / n_samples <= 0.3


class TestConfigValidation:
    """Test configuration validation for operation types."""

    def test_validate_valid_operation_types(self):
        """Valid operation type config should pass validation."""
        from endive.config import validate_config

        config = {
            'catalog': {'num_tables': 5},
            'transaction': {
                'operation_types': {
                    'fast_append': 0.7,
                    'merge_append': 0.2,
                    'validated_overwrite': 0.1,
                }
            }
        }

        errors, warnings = validate_config(config)

        assert len(errors) == 0, f"Should have no errors: {errors}"
        # May have warning about weights not summing to exactly 1.0

    def test_validate_invalid_operation_type_name(self):
        """Invalid operation type name should produce error."""
        from endive.config import validate_config

        config = {
            'catalog': {'num_tables': 5},
            'transaction': {
                'operation_types': {
                    'fast_append': 0.5,
                    'invalid_type': 0.5,  # Invalid!
                }
            }
        }

        errors, warnings = validate_config(config)

        assert len(errors) > 0
        assert any('invalid_type' in e for e in errors)

    def test_validate_negative_weight(self):
        """Negative weight should produce error."""
        from endive.config import validate_config

        config = {
            'catalog': {'num_tables': 5},
            'transaction': {
                'operation_types': {
                    'fast_append': 1.5,
                    'validated_overwrite': -0.5,  # Negative!
                }
            }
        }

        errors, warnings = validate_config(config)

        assert len(errors) > 0
        assert any('validated_overwrite' in e and '-0.5' in e for e in errors)

    def test_validate_warns_on_unnormalized_weights(self):
        """Weights not summing to 1.0 should produce warning."""
        from endive.config import validate_config

        config = {
            'catalog': {'num_tables': 5},
            'transaction': {
                'operation_types': {
                    'fast_append': 0.3,
                    'merge_append': 0.3,
                    # Sum = 0.6, not 1.0
                }
            }
        }

        errors, warnings = validate_config(config)

        assert len(errors) == 0, "Should not be an error"
        assert len(warnings) > 0
        assert any('normalized' in w.lower() for w in warnings)

    def test_validate_warns_about_validation_exceptions(self):
        """Should warn when validated_overwrite + real_conflict_probability will cause aborts."""
        from endive.config import validate_config

        config = {
            'catalog': {'num_tables': 5},
            'transaction': {
                'real_conflict_probability': 0.5,
                'operation_types': {
                    'validated_overwrite': 0.4,
                    'fast_append': 0.6,
                }
            }
        }

        errors, warnings = validate_config(config)

        assert len(errors) == 0
        assert any('abort' in w.lower() or 'validationexception' in w.lower() for w in warnings)


class TestPartitionAwareConflictResolution:
    """Tests for partition-aware conflict resolution with operation types."""

    def test_compute_overlapping_partitions(self):
        """Should correctly identify partitions with version changes."""
        from endive.conflict import _compute_overlapping_partitions
        from endive.transaction import Txn

        txn = Txn(
            id=1,
            t_submit=0,
            t_runtime=1000,
            v_catalog_seq=10,
            v_tblr={0: 10},
            v_tblw={0: 10},
        )
        # Txn read/wrote partitions 0 and 1
        txn.partitions_read = {0: {0, 1}}
        txn.partitions_written = {0: {0, 1}}
        # Txn saw version 5 for both partitions
        txn.v_partition_seq = {0: {0: 5, 1: 5}}

        # Partition 0 changed to v6, partition 1 unchanged
        partition_seq_snapshot = {0: {0: 6, 1: 5}}

        overlapping = _compute_overlapping_partitions(txn, partition_seq_snapshot)

        assert overlapping == {0: {0}}, "Only partition 0 should be overlapping"

    def test_partition_conflict_resolution_fast_append_always_retries(self):
        """FastAppend should always retry, never abort."""
        from endive.conflict import resolve_partition_conflict
        from endive.transaction import Txn
        from endive.snapshot import CatalogSnapshot
        from endive.operation import OperationType
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main
        import simpy

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                env = simpy.Environment()

                txn = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=1000,
                    v_catalog_seq=10,
                    v_tblr={0: 10},
                    v_tblw={0: 10},
                )
                txn.operation_type = OperationType.FAST_APPEND
                txn.partitions_read = {0: {0}}
                txn.partitions_written = {0: {0}}
                txn.v_partition_seq = {0: {0: 5}}

                snapshot = CatalogSnapshot(
                    seq=11,
                    tbl=(11,),
                    partition_seq=((6,),),
                    ml_offset=(1000,),
                    partition_ml_offset=((1000,),),
                    timestamp=1000,
                )

                partition_seq_snapshot = {0: {0: 6}}

                def run_resolution():
                    result = yield from resolve_partition_conflict(
                        env, txn, snapshot,
                        partition_seq_snapshot=partition_seq_snapshot,
                        data_overlap_probability=1.0,  # Would abort for validated
                        manifest_list_mode="rewrite",
                        manifests_per_commit=1.0,
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run_resolution())
                env.run()
                result = process.value

                # FastAppend should always retry
                assert result.should_retry is True
                assert result.abort_reason is None
                assert endive.main.STATS.false_conflicts == 1
                assert endive.main.STATS.real_conflicts == 0

            finally:
                os.unlink(config_path)

    def test_partition_conflict_resolution_validated_overwrite_aborts(self):
        """ValidatedOverwrite should abort on real conflict after reading ML."""
        from endive.conflict import resolve_partition_conflict
        from endive.transaction import Txn
        from endive.snapshot import CatalogSnapshot
        from endive.operation import OperationType
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main
        import simpy

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                np.random.seed(42)
                env = simpy.Environment()

                txn = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=1000,
                    v_catalog_seq=10,
                    v_tblr={0: 10},
                    v_tblw={0: 10},
                )
                txn.operation_type = OperationType.VALIDATED_OVERWRITE
                txn.partitions_read = {0: {0}}
                txn.partitions_written = {0: {0}}
                txn.v_partition_seq = {0: {0: 5}}

                snapshot = CatalogSnapshot(
                    seq=11,
                    tbl=(11,),
                    partition_seq=((6,),),
                    ml_offset=(1000,),
                    partition_ml_offset=((1000,),),
                    timestamp=1000,
                )

                partition_seq_snapshot = {0: {0: 6}}

                def run_resolution():
                    result = yield from resolve_partition_conflict(
                        env, txn, snapshot,
                        partition_seq_snapshot=partition_seq_snapshot,
                        data_overlap_probability=1.0,  # 100% real conflict
                        manifest_list_mode="rewrite",
                        manifests_per_commit=1.0,
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run_resolution())
                env.run()
                result = process.value

                # ValidatedOverwrite should abort on real conflict
                assert result.should_retry is False
                assert result.abort_reason == "validation_exception"
                assert endive.main.STATS.real_conflicts == 1
                assert endive.main.STATS.validation_exceptions == 1
                # ML read cost was paid BEFORE abort (key requirement)
                assert endive.main.STATS.manifest_list_reads == 1

            finally:
                os.unlink(config_path)

    def test_partition_conflict_pays_ml_read_before_abort(self):
        """Real conflict should pay ML read cost BEFORE aborting."""
        from endive.conflict import resolve_partition_conflict
        from endive.transaction import Txn
        from endive.snapshot import CatalogSnapshot
        from endive.operation import OperationType
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main
        import simpy

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                np.random.seed(42)
                env = simpy.Environment()

                # Transaction is 3 snapshots behind on partition 0
                txn = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=1000,
                    v_catalog_seq=10,
                    v_tblr={0: 10},
                    v_tblw={0: 10},
                )
                txn.operation_type = OperationType.VALIDATED_OVERWRITE
                txn.partitions_read = {0: {0}}
                txn.partitions_written = {0: {0}}
                txn.v_partition_seq = {0: {0: 5}}  # Txn at version 5

                snapshot = CatalogSnapshot(
                    seq=13,
                    tbl=(13,),
                    partition_seq=((8,),),  # Partition at version 8 (3 behind)
                    ml_offset=(1000,),
                    partition_ml_offset=((1000,),),
                    timestamp=1000,
                )

                partition_seq_snapshot = {0: {0: 8}}  # 8 - 5 = 3 snapshots behind

                def run_resolution():
                    result = yield from resolve_partition_conflict(
                        env, txn, snapshot,
                        partition_seq_snapshot=partition_seq_snapshot,
                        data_overlap_probability=1.0,
                        manifest_list_mode="rewrite",
                        manifests_per_commit=1.0,
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run_resolution())
                env.run()
                result = process.value

                # Should abort but AFTER paying ML read cost
                assert result.should_retry is False
                # Paid 3 ML reads (one per snapshot behind) BEFORE aborting
                assert endive.main.STATS.manifest_list_reads == 3

            finally:
                os.unlink(config_path)

    def test_data_overlap_probability_config(self):
        """DATA_OVERLAP_PROBABILITY should be loadable from config."""
        import endive.main as main_module

        # Use the same format as create_test_config
        toml_content = """
[simulation]
duration_ms = 10000
output_path = "test.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 10
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
data_overlap_probability = 0.42

inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0
inter_arrival.min = 100.0
inter_arrival.max = 1000.0
inter_arrival.mean = 500.0
inter_arrival.std_dev = 100.0
inter_arrival.value = 500.0

ntable.zipf = 2.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[partition]
enabled = true
num_partitions = 4
partitions_per_txn_mean = 2
partitions_per_txn_max = 3

[storage]
max_parallel = 4
min_latency = 5

T_CAS.mean = 100
T_CAS.stddev = 10

T_METADATA_ROOT.read.mean = 50
T_METADATA_ROOT.read.stddev = 5
T_METADATA_ROOT.write.mean = 60
T_METADATA_ROOT.write.stddev = 6

T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6

T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            main_module.configure_from_toml(config_path)
            assert main_module.DATA_OVERLAP_PROBABILITY == 0.42
        finally:
            os.unlink(config_path)

    def test_data_overlap_probability_defaults_to_real_conflict(self):
        """DATA_OVERLAP_PROBABILITY should default to REAL_CONFLICT_PROBABILITY."""
        import endive.main as main_module

        toml_content = """
[simulation]
duration_ms = 10000
output_path = "test.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 10
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5
real_conflict_probability = 0.35
# data_overlap_probability not set - should default to real_conflict_probability

inter_arrival.distribution = "exponential"
inter_arrival.scale = 500.0
inter_arrival.min = 100.0
inter_arrival.max = 1000.0
inter_arrival.mean = 500.0
inter_arrival.std_dev = 100.0
inter_arrival.value = 500.0

ntable.zipf = 2.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[partition]
enabled = true
num_partitions = 4

[storage]
max_parallel = 4
min_latency = 5

T_CAS.mean = 100
T_CAS.stddev = 10

T_METADATA_ROOT.read.mean = 50
T_METADATA_ROOT.read.stddev = 5
T_METADATA_ROOT.write.mean = 60
T_METADATA_ROOT.write.stddev = 6

T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6

T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            f.write(toml_content)
            config_path = f.name

        try:
            main_module.configure_from_toml(config_path)
            # Should default to REAL_CONFLICT_PROBABILITY
            assert main_module.DATA_OVERLAP_PROBABILITY == 0.35
        finally:
            os.unlink(config_path)

    def test_multiple_partitions_stops_on_first_real_conflict(self):
        """With multiple overlapping partitions, should abort on first real conflict."""
        from endive.conflict import resolve_partition_conflict
        from endive.transaction import Txn
        from endive.snapshot import CatalogSnapshot
        from endive.operation import OperationType
        from endive.main import configure_from_toml
        from endive.test_utils import create_test_config
        import endive.main
        import simpy

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=1000,
            )

            try:
                configure_from_toml(config_path)
                endive.main.STATS = Stats()

                np.random.seed(42)
                env = simpy.Environment()

                # Transaction with 3 overlapping partitions
                txn = Txn(
                    id=1,
                    t_submit=0,
                    t_runtime=1000,
                    v_catalog_seq=10,
                    v_tblr={0: 10},
                    v_tblw={0: 10},
                )
                txn.operation_type = OperationType.VALIDATED_OVERWRITE
                txn.partitions_written = {0: {0, 1, 2}}
                txn.partitions_read = {}
                txn.v_partition_seq = {0: {0: 5, 1: 5, 2: 5}}

                # All 3 partitions have changed
                snapshot = CatalogSnapshot(
                    seq=11,
                    tbl=(11,),
                    partition_seq=((6, 6, 6),),
                    ml_offset=(1000,),
                    partition_ml_offset=((1000, 1000, 1000),),
                    timestamp=1000,
                )

                partition_seq_snapshot = {0: {0: 6, 1: 6, 2: 6}}

                def run_resolution():
                    result = yield from resolve_partition_conflict(
                        env, txn, snapshot,
                        partition_seq_snapshot=partition_seq_snapshot,
                        data_overlap_probability=1.0,  # Real conflict
                        manifest_list_mode="rewrite",
                        manifests_per_commit=1.0,
                        stats=endive.main.STATS,
                    )
                    return result

                process = env.process(run_resolution())
                env.run()
                result = process.value

                # Should abort after first real conflict
                assert result.should_retry is False
                assert result.abort_reason == "validation_exception"
                # Only paid ML read for first partition before aborting
                assert endive.main.STATS.manifest_list_reads == 1
                assert endive.main.STATS.real_conflicts == 1

            finally:
                os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
