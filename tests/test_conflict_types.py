"""Tests for false vs real conflict resolution functionality.

Verifies that:
1. False conflicts (version changed, no data overlap) skip manifest file operations
2. Real conflicts (overlapping data) trigger manifest file read/write operations
3. Conflicting manifests distribution sampling works correctly
4. Statistics correctly track false vs real conflicts
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from icecap.main import configure_from_toml
import icecap.main
from icecap.capstats import Stats
from icecap.test_utils import create_test_config


class TestFalseConflicts:
    """Test that false conflicts behave correctly."""

    def test_false_conflicts_only(self):
        """Verify false conflicts don't trigger manifest file operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path=os.path.join(tmpdir, "test.parquet"),
                seed=42,
                duration_ms=5000,
                inter_arrival_scale=50.0,  # High contention
                num_tables=1
            )

            # Add real_conflict_probability = 0.0 to config
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
                icecap.main.TABLE_TO_GROUP = {i: 0 for i in range(icecap.main.N_TABLES)}
                icecap.main.GROUP_TO_TABLES = {0: list(range(icecap.main.N_TABLES))}
                icecap.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(icecap.main.setup(env))
                env.run(until=icecap.main.SIM_DURATION_MS)

                # Verify only false conflicts occurred
                assert icecap.main.STATS.false_conflicts > 0, "Should have some false conflicts"
                assert icecap.main.STATS.real_conflicts == 0, "Should have no real conflicts"
                assert icecap.main.STATS.manifest_files_read == 0, "Should not read manifest files"
                assert icecap.main.STATS.manifest_files_written == 0, "Should not write manifest files"

                print(f"✓ False conflicts only test passed")
                print(f"  False conflicts: {icecap.main.STATS.false_conflicts}")
                print(f"  Real conflicts: {icecap.main.STATS.real_conflicts}")

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
                icecap.main.TABLE_TO_GROUP = {i: 0 for i in range(icecap.main.N_TABLES)}
                icecap.main.GROUP_TO_TABLES = {0: list(range(icecap.main.N_TABLES))}
                icecap.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(icecap.main.setup(env))
                env.run(until=icecap.main.SIM_DURATION_MS)

                # Verify only real conflicts occurred
                assert icecap.main.STATS.real_conflicts > 0, "Should have some real conflicts"
                assert icecap.main.STATS.false_conflicts == 0, "Should have no false conflicts"
                assert icecap.main.STATS.manifest_files_read > 0, "Should read manifest files"
                assert icecap.main.STATS.manifest_files_written > 0, "Should write manifest files"

                # Verify approximately 3 manifest files per real conflict (fixed distribution)
                avg_read = icecap.main.STATS.manifest_files_read / icecap.main.STATS.real_conflicts
                avg_written = icecap.main.STATS.manifest_files_written / icecap.main.STATS.real_conflicts

                assert 2.5 <= avg_read <= 3.5, f"Expected ~3 manifest files read per conflict, got {avg_read:.1f}"
                assert 2.5 <= avg_written <= 3.5, f"Expected ~3 manifest files written per conflict, got {avg_written:.1f}"

                print(f"✓ Real conflicts only test passed")
                print(f"  False conflicts: {icecap.main.STATS.false_conflicts}")
                print(f"  Real conflicts: {icecap.main.STATS.real_conflicts}")
                print(f"  Manifest files read: {icecap.main.STATS.manifest_files_read} (avg {avg_read:.1f})")
                print(f"  Manifest files written: {icecap.main.STATS.manifest_files_written} (avg {avg_written:.1f})")

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
                icecap.main.TABLE_TO_GROUP = {i: 0 for i in range(icecap.main.N_TABLES)}
                icecap.main.GROUP_TO_TABLES = {0: list(range(icecap.main.N_TABLES))}
                icecap.main.STATS = Stats()

                # Run simulation
                import simpy
                env = simpy.Environment()
                env.process(icecap.main.setup(env))
                env.run(until=icecap.main.SIM_DURATION_MS)

                # Verify both types of conflicts occurred
                assert icecap.main.STATS.false_conflicts > 0, "Should have some false conflicts"
                assert icecap.main.STATS.real_conflicts > 0, "Should have some real conflicts"

                # Verify ratio is approximately 50/50 (within reasonable tolerance)
                total = icecap.main.STATS.false_conflicts + icecap.main.STATS.real_conflicts
                false_ratio = icecap.main.STATS.false_conflicts / total
                real_ratio = icecap.main.STATS.real_conflicts / total

                # Allow 20% deviation from expected 0.5
                assert 0.3 <= false_ratio <= 0.7, f"Expected ~50% false conflicts, got {false_ratio*100:.1f}%"
                assert 0.3 <= real_ratio <= 0.7, f"Expected ~50% real conflicts, got {real_ratio*100:.1f}%"

                print(f"✓ Mixed conflicts test passed")
                print(f"  False conflicts: {icecap.main.STATS.false_conflicts} ({false_ratio*100:.1f}%)")
                print(f"  Real conflicts: {icecap.main.STATS.real_conflicts} ({real_ratio*100:.1f}%)")

            finally:
                os.unlink(config_path)


class TestConflictingManifestsDistribution:
    """Test conflicting manifests sampling distributions."""

    def test_fixed_distribution(self):
        """Verify fixed distribution returns constant value."""
        # Save current config
        old_dist = icecap.main.CONFLICTING_MANIFESTS_DIST
        old_params = icecap.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            icecap.main.CONFLICTING_MANIFESTS_DIST = "fixed"
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = {"value": 5}

            # Sample many times
            samples = [icecap.main.sample_conflicting_manifests() for _ in range(100)]

            # All should be exactly 5
            assert all(s == 5 for s in samples), "Fixed distribution should always return same value"

            print(f"✓ Fixed distribution test passed")
            print(f"  All 100 samples = 5")

        finally:
            icecap.main.CONFLICTING_MANIFESTS_DIST = old_dist
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = old_params

    def test_uniform_distribution(self):
        """Verify uniform distribution respects min/max bounds."""
        old_dist = icecap.main.CONFLICTING_MANIFESTS_DIST
        old_params = icecap.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            np.random.seed(42)
            icecap.main.CONFLICTING_MANIFESTS_DIST = "uniform"
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = {"min": 1, "max": 10}

            # Sample many times
            samples = [icecap.main.sample_conflicting_manifests() for _ in range(1000)]

            # All should be in range [1, 10]
            assert all(1 <= s <= 10 for s in samples), "Uniform distribution should respect bounds"

            # Should see values across the range
            assert len(set(samples)) >= 8, "Uniform distribution should produce varied values"

            print(f"✓ Uniform distribution test passed")
            print(f"  Min: {min(samples)}, Max: {max(samples)}, Unique values: {len(set(samples))}")

        finally:
            icecap.main.CONFLICTING_MANIFESTS_DIST = old_dist
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = old_params

    def test_exponential_distribution(self):
        """Verify exponential distribution respects bounds and mean."""
        old_dist = icecap.main.CONFLICTING_MANIFESTS_DIST
        old_params = icecap.main.CONFLICTING_MANIFESTS_PARAMS

        try:
            np.random.seed(42)
            icecap.main.CONFLICTING_MANIFESTS_DIST = "exponential"
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = {"mean": 3.0, "min": 1, "max": 10}

            # Sample many times
            samples = [icecap.main.sample_conflicting_manifests() for _ in range(1000)]

            # All should be in range [1, 10]
            assert all(1 <= s <= 10 for s in samples), "Exponential distribution should respect bounds"

            # Mean should be approximately 3.0 (before clamping)
            # After clamping, might be slightly higher
            sample_mean = np.mean(samples)
            assert 2.0 <= sample_mean <= 4.5, f"Expected mean ~3.0, got {sample_mean:.1f}"

            print(f"✓ Exponential distribution test passed")
            print(f"  Min: {min(samples)}, Max: {max(samples)}, Mean: {sample_mean:.1f}")

        finally:
            icecap.main.CONFLICTING_MANIFESTS_DIST = old_dist
            icecap.main.CONFLICTING_MANIFESTS_PARAMS = old_params


class TestConflictLatencyDifference:
    """Test that real conflicts have higher latency than false conflicts."""

    def test_real_conflicts_slower_than_false(self):
        """Verify real conflicts take longer to resolve than false conflicts."""
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
                prob = 0.0 if conflict_type == 'false' else 1.0

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
                    icecap.main.TABLE_TO_GROUP = {i: 0 for i in range(icecap.main.N_TABLES)}
                    icecap.main.GROUP_TO_TABLES = {0: list(range(icecap.main.N_TABLES))}
                    icecap.main.STATS = Stats()

                    # Run simulation
                    import simpy
                    env = simpy.Environment()
                    env.process(icecap.main.setup(env))
                    env.run(until=icecap.main.SIM_DURATION_MS)

                    # Get average latency for transactions with retries
                    df = pd.DataFrame(icecap.main.STATS.transactions)
                    with_retries = df[(df['status'] == 'committed') & (df['n_retries'] > 0)]

                    if len(with_retries) > 0:
                        results[conflict_type] = with_retries['commit_latency'].mean()

                finally:
                    os.unlink(config_path)

        # Verify real conflicts have higher latency
        if 'false' in results and 'real' in results:
            assert results['real'] > results['false'], \
                f"Real conflicts ({results['real']:.1f}ms) should be slower than false conflicts ({results['false']:.1f}ms)"

            print(f"✓ Conflict latency difference test passed")
            print(f"  False conflict avg latency: {results['false']:.1f}ms")
            print(f"  Real conflict avg latency: {results['real']:.1f}ms")
            print(f"  Difference: {results['real'] - results['false']:.1f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
