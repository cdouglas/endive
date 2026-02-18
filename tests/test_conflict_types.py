"""Tests for false vs real conflict resolution functionality.

Verifies that:
1. False conflicts (same table, no data overlap) require ML operations in rewrite mode
2. False conflicts skip ML operations in ML+ mode (tentative entry still valid)
3. Real conflicts (overlapping data) trigger manifest file read/write operations
4. Conflicting manifests distribution sampling works correctly
5. Statistics correctly track false vs real conflicts and ML operations
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from endive.main import configure_from_toml
import endive.main
from endive.capstats import Stats
from endive.test_utils import create_test_config


class TestFalseConflicts:
    """Test that false conflicts behave correctly."""

    def test_false_conflicts_rewrite_mode(self):
        """Verify false conflicts in rewrite mode require ML operations but not manifest file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=5000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1
            )

            # Add real_conflict_probability = 0.0 to config (default is rewrite mode)
            with open(config_path, 'r') as f:
                content = f.read()

            content = content.replace(
                'seltblw.zipf = 1.2',
                'seltblw.zipf = 1.2\nreal_conflict_probability = 0.0'
            )

            with open(config_path, 'w') as f:
                f.write(content)

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.TABLE_TO_GROUP = {i: 0 for i in range(endive.main.N_TABLES)}
                endive.main.GROUP_TO_TABLES = {0: list(range(endive.main.N_TABLES))}
                endive.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify only false conflicts occurred
                assert endive.main.STATS.false_conflicts > 0, "Should have some false conflicts"
                assert endive.main.STATS.real_conflicts == 0, "Should have no real conflicts"

                # False conflicts should NOT trigger manifest FILE operations
                assert endive.main.STATS.manifest_files_read == 0, "Should not read manifest files"
                assert endive.main.STATS.manifest_files_written == 0, "Should not write manifest files"

                # In rewrite mode, false conflicts SHOULD trigger manifest LIST operations:
                # 1. History reads: N MLs for N snapshots behind (like Iceberg's validationHistory)
                # 2. Resolution reads/writes: 1 ML per conflict (rewrite with combined pointers)
                # Note: History reads can be much larger than conflict count under high contention
                # because transactions may be many snapshots behind when they retry
                total_conflicts = endive.main.STATS.false_conflicts + endive.main.STATS.real_conflicts
                assert endive.main.STATS.manifest_list_reads >= endive.main.STATS.false_conflicts, \
                    f"Expected at least {endive.main.STATS.false_conflicts} ML reads, got {endive.main.STATS.manifest_list_reads}"
                assert endive.main.STATS.manifest_list_writes >= endive.main.STATS.false_conflicts, \
                    f"Expected at least {endive.main.STATS.false_conflicts} ML writes, got {endive.main.STATS.manifest_list_writes}"
                # ML reads include history + resolution; no upper bound since history can be large

                print(f"✓ False conflicts (rewrite mode) test passed")
                print(f"  False conflicts: {endive.main.STATS.false_conflicts}")
                print(f"  Real conflicts: {endive.main.STATS.real_conflicts}")
                print(f"  Manifest list reads: {endive.main.STATS.manifest_list_reads}")
                print(f"  Manifest list writes: {endive.main.STATS.manifest_list_writes}")
                print(f"  Manifest file operations: read={endive.main.STATS.manifest_files_read}, write={endive.main.STATS.manifest_files_written}")

            finally:
                os.unlink(config_path)

    def test_false_conflicts_ml_append_mode(self):
        """Verify false conflicts in ML+ mode skip ML operations (tentative entry still valid)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=5000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1
            )

            # Add ML+ mode and real_conflict_probability = 0.0
            with open(config_path, 'r') as f:
                content = f.read()

            content = content.replace(
                'seltblw.zipf = 1.2',
                'seltblw.zipf = 1.2\nreal_conflict_probability = 0.0\nmanifest_list_mode = "append"'
            )

            with open(config_path, 'w') as f:
                f.write(content)

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.TABLE_TO_GROUP = {i: 0 for i in range(endive.main.N_TABLES)}
                endive.main.GROUP_TO_TABLES = {0: list(range(endive.main.N_TABLES))}
                endive.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify only false conflicts occurred
                assert endive.main.STATS.false_conflicts > 0, "Should have some false conflicts"
                assert endive.main.STATS.real_conflicts == 0, "Should have no real conflicts"

                # False conflicts should NOT trigger manifest FILE operations
                assert endive.main.STATS.manifest_files_read == 0, "Should not read manifest files"
                assert endive.main.STATS.manifest_files_written == 0, "Should not write manifest files"

                # In ML+ mode, false conflicts skip ML read/write in RESOLUTION
                # (tentative entry is still valid - readers filter by committed txn list)
                # BUT we still read ML HISTORY for conflict detection (like Iceberg's validationHistory)
                # So ML reads > 0 (history), but ML writes == 0 (skipped in resolution)
                assert endive.main.STATS.manifest_list_reads > 0, \
                    f"Should have ML history reads, got {endive.main.STATS.manifest_list_reads}"
                assert endive.main.STATS.manifest_list_writes == 0, \
                    f"ML+ mode should skip ML writes on false conflicts, got {endive.main.STATS.manifest_list_writes}"

                print(f"✓ False conflicts (ML+ mode) test passed")
                print(f"  False conflicts: {endive.main.STATS.false_conflicts}")
                print(f"  Manifest list reads (history): {endive.main.STATS.manifest_list_reads}")
                print(f"  Manifest list writes: 0 (tentative entries still valid)")

            finally:
                os.unlink(config_path)


class TestRealConflicts:
    """Test that real conflicts behave correctly."""

    def test_real_conflicts_only(self):
        """Verify real conflicts trigger manifest file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=5000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1
            )

            # Add real_conflict_probability = 1.0 to config
            with open(config_path, 'r') as f:
                content = f.read()

            # Insert into [transaction] section
            content = content.replace(
                'seltblw.zipf = 1.2',
                'seltblw.zipf = 1.2\nreal_conflict_probability = 1.0\n' +
                'conflicting_manifests.distribution = "fixed"\n' +
                'conflicting_manifests.value = 3'
            )

            with open(config_path, 'w') as f:
                f.write(content)

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.TABLE_TO_GROUP = {i: 0 for i in range(endive.main.N_TABLES)}
                endive.main.GROUP_TO_TABLES = {0: list(range(endive.main.N_TABLES))}
                endive.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify only real conflicts occurred
                assert endive.main.STATS.real_conflicts > 0, "Should have some real conflicts"
                assert endive.main.STATS.false_conflicts == 0, "Should have no false conflicts"
                assert endive.main.STATS.manifest_files_read > 0, "Should read manifest files"
                assert endive.main.STATS.manifest_files_written > 0, "Should write manifest files"

                # Verify approximately 3 manifest files per real conflict (fixed distribution)
                avg_read = endive.main.STATS.manifest_files_read / endive.main.STATS.real_conflicts
                avg_written = endive.main.STATS.manifest_files_written / endive.main.STATS.real_conflicts

                assert 2.5 <= avg_read <= 3.5, f"Expected ~3 manifest files read per conflict, got {avg_read:.1f}"
                assert 2.5 <= avg_written <= 3.5, f"Expected ~3 manifest files written per conflict, got {avg_written:.1f}"

                print(f"✓ Real conflicts only test passed")
                print(f"  False conflicts: {endive.main.STATS.false_conflicts}")
                print(f"  Real conflicts: {endive.main.STATS.real_conflicts}")
                print(f"  Manifest files read: {endive.main.STATS.manifest_files_read} (avg {avg_read:.1f})")
                print(f"  Manifest files written: {endive.main.STATS.manifest_files_written} (avg {avg_written:.1f})")

            finally:
                os.unlink(config_path)

    def test_mixed_conflicts(self):
        """Verify mixed false/real conflicts produce expected ratio."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=5000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1
            )

            # Add real_conflict_probability = 0.5 to config
            with open(config_path, 'r') as f:
                content = f.read()

            content = content.replace(
                'seltblw.zipf = 1.2',
                'seltblw.zipf = 1.2\nreal_conflict_probability = 0.5'
            )

            with open(config_path, 'w') as f:
                f.write(content)

            try:
                configure_from_toml(config_path)
                np.random.seed(42)
                endive.main.TABLE_TO_GROUP = {i: 0 for i in range(endive.main.N_TABLES)}
                endive.main.GROUP_TO_TABLES = {0: list(range(endive.main.N_TABLES))}
                endive.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(endive.main.setup(env))
                env.run(until=endive.main.SIM_DURATION_MS)

                # Verify both types of conflicts occurred
                assert endive.main.STATS.false_conflicts > 0, "Should have some false conflicts"
                assert endive.main.STATS.real_conflicts > 0, "Should have some real conflicts"

                # Verify ratio is approximately 50/50 (within reasonable tolerance)
                total = endive.main.STATS.false_conflicts + endive.main.STATS.real_conflicts
                false_ratio = endive.main.STATS.false_conflicts / total
                real_ratio = endive.main.STATS.real_conflicts / total

                # Allow 20% deviation from expected 0.5
                assert 0.3 <= false_ratio <= 0.7, f"Expected ~50% false conflicts, got {false_ratio*100:.1f}%"
                assert 0.3 <= real_ratio <= 0.7, f"Expected ~50% real conflicts, got {real_ratio*100:.1f}%"

                print(f"✓ Mixed conflicts test passed")
                print(f"  False conflicts: {endive.main.STATS.false_conflicts} ({false_ratio*100:.1f}%)")
                print(f"  Real conflicts: {endive.main.STATS.real_conflicts} ({real_ratio*100:.1f}%)")

            finally:
                os.unlink(config_path)


class TestConflictingManifestsDistribution:
    """Test conflicting manifests sampling distributions."""

    def test_fixed_distribution(self):
        """Verify fixed distribution returns constant value."""
        # Save current config
        old_dist = endive.main.CONFLICTING_MANIFESTS_DIST
        old_params = endive.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            endive.main.CONFLICTING_MANIFESTS_DIST = "fixed"
            endive.main.CONFLICTING_MANIFESTS_PARAMS = {"value": 5}

            # Sample many times
            samples = [endive.main.sample_conflicting_manifests() for _ in range(100)]

            # All should be exactly 5
            assert all(s == 5 for s in samples), "Fixed distribution should always return same value"

            print(f"✓ Fixed distribution test passed")
            print(f"  All 100 samples = 5")

        finally:
            endive.main.CONFLICTING_MANIFESTS_DIST = old_dist
            endive.main.CONFLICTING_MANIFESTS_PARAMS = old_params

    def test_uniform_distribution(self):
        """Verify uniform distribution respects min/max bounds."""
        old_dist = endive.main.CONFLICTING_MANIFESTS_DIST
        old_params = endive.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            np.random.seed(42)
            endive.main.CONFLICTING_MANIFESTS_DIST = "uniform"
            endive.main.CONFLICTING_MANIFESTS_PARAMS = {"min": 1, "max": 10}

            # Sample many times
            samples = [endive.main.sample_conflicting_manifests() for _ in range(1000)]

            # All should be in range [1, 10]
            assert all(1 <= s <= 10 for s in samples), "Uniform distribution should respect bounds"

            # Should see values across the range
            assert len(set(samples)) >= 8, "Uniform distribution should produce varied values"

            print(f"✓ Uniform distribution test passed")
            print(f"  Min: {min(samples)}, Max: {max(samples)}, Unique values: {len(set(samples))}")

        finally:
            endive.main.CONFLICTING_MANIFESTS_DIST = old_dist
            endive.main.CONFLICTING_MANIFESTS_PARAMS = old_params

    def test_exponential_distribution(self):
        """Verify exponential distribution respects bounds and mean."""
        old_dist = endive.main.CONFLICTING_MANIFESTS_DIST
        old_params = endive.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            np.random.seed(42)
            endive.main.CONFLICTING_MANIFESTS_DIST = "exponential"
            endive.main.CONFLICTING_MANIFESTS_PARAMS = {"mean": 3.0, "min": 1, "max": 10}

            # Sample many times
            samples = [endive.main.sample_conflicting_manifests() for _ in range(1000)]

            # All should be in range [1, 10]
            assert all(1 <= s <= 10 for s in samples), "Exponential distribution should respect bounds"

            # Mean should be approximately 3.0 (before clamping)
            # After clamping, might be slightly higher
            sample_mean = np.mean(samples)
            assert 2.0 <= sample_mean <= 4.5, f"Expected mean ~3.0, got {sample_mean:.1f}"

            print(f"✓ Exponential distribution test passed")
            print(f"  Min: {min(samples)}, Max: {max(samples)}, Mean: {sample_mean:.1f}")

        finally:
            endive.main.CONFLICTING_MANIFESTS_DIST = old_dist
            endive.main.CONFLICTING_MANIFESTS_PARAMS = old_params


class TestConflictLatencyDifference:
    """Test that real conflicts have more I/O operations than false conflicts."""

    def test_real_conflicts_have_more_io_operations(self):
        """Verify real conflicts require more I/O operations than false conflicts.

        Both false and real conflicts require manifest list operations (in rewrite mode).
        Real conflicts additionally require manifest file read/write operations.
        """
        results = {}

        for conflict_type in ['false', 'real']:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = create_test_config(
                    output_path=os.path.join(tmpdir, f"test_{conflict_type}.parquet"),
                    seed=42,
                    duration_ms=5000,
                    inter_arrival_scale=50.0,
                    num_tables=1
                )

                # Configure conflict type
                with open(config_path, 'r') as f:
                    content = f.read()

                if conflict_type == 'real':
                    content = content.replace(
                        'seltblw.zipf = 1.2',
                        'seltblw.zipf = 1.2\nreal_conflict_probability = 1.0\n' +
                        'conflicting_manifests.distribution = "fixed"\n' +
                        'conflicting_manifests.value = 3'
                    )
                else:
                    content = content.replace(
                        'seltblw.zipf = 1.2',
                        'seltblw.zipf = 1.2\nreal_conflict_probability = 0.0'
                    )

                with open(config_path, 'w') as f:
                    f.write(content)

                try:
                    configure_from_toml(config_path)
                    np.random.seed(42)
                    endive.main.TABLE_TO_GROUP = {i: 0 for i in range(endive.main.N_TABLES)}
                    endive.main.GROUP_TO_TABLES = {0: list(range(endive.main.N_TABLES))}
                    endive.main.STATS = Stats()

                    # Run simulation
                    import simpy
                    env = simpy.Environment()
                    env.process(endive.main.setup(env))
                    env.run(until=endive.main.SIM_DURATION_MS)

                    # Collect I/O operation counts
                    results[conflict_type] = {
                        'conflicts': endive.main.STATS.false_conflicts + endive.main.STATS.real_conflicts,
                        'ml_reads': endive.main.STATS.manifest_list_reads,
                        'ml_writes': endive.main.STATS.manifest_list_writes,
                        'mf_reads': endive.main.STATS.manifest_files_read,
                        'mf_writes': endive.main.STATS.manifest_files_written,
                    }

                finally:
                    os.unlink(config_path)

        # Verify both have ML operations (in rewrite mode)
        assert results['false']['ml_reads'] > 0, "False conflicts should have ML reads"
        assert results['false']['ml_writes'] > 0, "False conflicts should have ML writes"
        assert results['real']['ml_reads'] > 0, "Real conflicts should have ML reads"
        assert results['real']['ml_writes'] > 0, "Real conflicts should have ML writes"

        # Verify false conflicts have NO manifest file operations
        assert results['false']['mf_reads'] == 0, "False conflicts should not read manifest files"
        assert results['false']['mf_writes'] == 0, "False conflicts should not write manifest files"

        # Verify real conflicts have manifest file operations
        assert results['real']['mf_reads'] > 0, "Real conflicts should read manifest files"
        assert results['real']['mf_writes'] > 0, "Real conflicts should write manifest files"

        # Total I/O for real conflicts should be higher
        false_io = results['false']['ml_reads'] + results['false']['ml_writes']
        real_io = (results['real']['ml_reads'] + results['real']['ml_writes'] +
                   results['real']['mf_reads'] + results['real']['mf_writes'])

        assert real_io > false_io, \
            f"Real conflicts should have more I/O operations ({real_io}) than false ({false_io})"

        print(f"✓ Conflict I/O operations test passed")
        print(f"  False conflicts: {results['false']['conflicts']}")
        print(f"    ML reads/writes: {results['false']['ml_reads']}/{results['false']['ml_writes']}")
        print(f"    MF reads/writes: {results['false']['mf_reads']}/{results['false']['mf_writes']}")
        print(f"  Real conflicts: {results['real']['conflicts']}")
        print(f"    ML reads/writes: {results['real']['ml_reads']}/{results['real']['ml_writes']}")
        print(f"    MF reads/writes: {results['real']['mf_reads']}/{results['real']['mf_writes']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
