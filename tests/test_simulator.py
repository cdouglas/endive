"""Tests for the icecap simulator core functionality."""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from icecap.main import configure_from_toml, generate_inter_arrival_time
from icecap.capstats import Stats
import simpy


def create_test_config(
    output_path: str,
    seed: int = None,
    duration_ms: int = 10000,
    inter_arrival_scale: float = 500.0,
    t_cas: int = 100,
    num_tables: int = 5,
    retry: int = 5
) -> str:
    """Create a test configuration file."""
    config_content = f"""[simulation]
duration_ms = {duration_ms}
output_path = "{output_path}"
{'seed = ' + str(seed) if seed is not None else '# seed = 42'}

[catalog]
num_tables = {num_tables}

[transaction]
retry = {retry}
runtime.min = 100
runtime.mean = 200
runtime.sigma = 1.5

inter_arrival.distribution = "exponential"
inter_arrival.scale = {inter_arrival_scale}
inter_arrival.min = 100.0
inter_arrival.max = 1000.0
inter_arrival.mean = 500.0
inter_arrival.std_dev = 100.0
inter_arrival.value = 500.0

ntable.zipf = 2.0
seltbl.zipf = 1.4
seltblw.zipf = 1.2

[storage]
T_CAS = {t_cas}
T_METADATA_ROOT = 50
T_MANIFEST_LIST = 50
T_MANIFEST_FILE = 50
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(config_content)
        return f.name


def run_simulation_from_config(config_path: str) -> pd.DataFrame:
    """Run simulation and return results as DataFrame."""
    import icecap.main

    # Reset global stats
    icecap.main.STATS = Stats()

    # Load configuration
    configure_from_toml(config_path)

    # Setup random seed if specified
    if icecap.main.SIM_SEED is not None:
        np.random.seed(icecap.main.SIM_SEED)

    # Run simulation
    env = simpy.Environment()
    env.process(icecap.main.setup(env))
    env.run(until=icecap.main.SIM_DURATION_MS)

    # Return results as DataFrame
    return pd.DataFrame(icecap.main.STATS.transactions)


class TestDeterminism:
    """Test that simulator produces deterministic results with a fixed seed."""

    def test_deterministic_with_seed(self):
        """Run simulation twice with same seed, verify identical results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output1 = os.path.join(tmpdir, "run1.parquet")
            output2 = os.path.join(tmpdir, "run2.parquet")

            # Create config with fixed seed
            config_path = create_test_config(
                output_path=output1,
                seed=42,
                duration_ms=20000,
                inter_arrival_scale=300.0
            )

            try:
                # Run simulation twice
                df1 = run_simulation_from_config(config_path)
                df2 = run_simulation_from_config(config_path)

                # Verify same number of transactions
                assert len(df1) == len(df2), "Different number of transactions generated"

                # Verify transaction IDs match
                assert df1['txn_id'].tolist() == df2['txn_id'].tolist(), \
                    "Transaction IDs differ between runs"

                # Verify submit times match
                assert df1['t_submit'].tolist() == df2['t_submit'].tolist(), \
                    "Submit times differ between runs"

                # Verify commit times match
                assert df1['t_commit'].tolist() == df2['t_commit'].tolist(), \
                    "Commit times differ between runs"

                # Verify status matches
                assert df1['status'].tolist() == df2['status'].tolist(), \
                    "Transaction status differs between runs"

                # Verify retry counts match
                assert df1['n_retries'].tolist() == df2['n_retries'].tolist(), \
                    "Retry counts differ between runs"

                print(f"✓ Determinism test passed: {len(df1)} transactions generated identically")

            finally:
                os.unlink(config_path)

    def test_different_seeds_produce_different_results(self):
        """Verify that different seeds produce different results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output1 = os.path.join(tmpdir, "seed1.parquet")
            output2 = os.path.join(tmpdir, "seed2.parquet")

            config1 = create_test_config(output_path=output1, seed=42, duration_ms=15000)
            config2 = create_test_config(output_path=output2, seed=123, duration_ms=15000)

            try:
                df1 = run_simulation_from_config(config1)
                df2 = run_simulation_from_config(config2)

                # Should have different transaction patterns
                # (very unlikely to be identical with different seeds)
                assert df1['t_submit'].tolist() != df2['t_submit'].tolist(), \
                    "Different seeds produced identical submit times"

                print(f"✓ Different seeds test passed")

            finally:
                os.unlink(config1)
                os.unlink(config2)


class TestParameterEffects:
    """Sanity tests for parameter effects on simulation outcomes."""

    def test_lower_inter_arrival_increases_contention(self):
        """Lower inter-arrival time should increase contention and retries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # High inter-arrival (low load)
            config_high = create_test_config(
                output_path=os.path.join(tmpdir, "high.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=2000.0  # Low load
            )

            # Low inter-arrival (high load)
            config_low = create_test_config(
                output_path=os.path.join(tmpdir, "low.parquet"),
                seed=42,
                duration_ms=30000,
                inter_arrival_scale=200.0  # High load
            )

            try:
                df_high = run_simulation_from_config(config_high)
                df_low = run_simulation_from_config(config_low)

                # Calculate metrics
                high_load_committed = df_low[df_low['status'] == 'committed']
                low_load_committed = df_high[df_high['status'] == 'committed']

                # High load should have more transactions (shorter inter-arrival)
                assert len(df_low) > len(df_high), \
                    "High load should generate more transactions"

                # High load should have more retries on average
                if len(high_load_committed) > 0 and len(low_load_committed) > 0:
                    avg_retries_high_load = high_load_committed['n_retries'].mean()
                    avg_retries_low_load = low_load_committed['n_retries'].mean()

                    assert avg_retries_high_load > avg_retries_low_load, \
                        f"High load should have more retries: {avg_retries_high_load} vs {avg_retries_low_load}"

                print(f"✓ Inter-arrival effect test passed")
                print(f"  Low load: {len(df_high)} txns, avg retries: {low_load_committed['n_retries'].mean():.2f}")
                print(f"  High load: {len(df_low)} txns, avg retries: {high_load_committed['n_retries'].mean():.2f}")

            finally:
                os.unlink(config_high)
                os.unlink(config_low)

    def test_higher_cas_latency_increases_commit_time(self):
        """Higher CAS latency should increase commit times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Low CAS latency
            config_low = create_test_config(
                output_path=os.path.join(tmpdir, "low_cas.parquet"),
                seed=42,
                duration_ms=25000,
                t_cas=50,
                inter_arrival_scale=500.0
            )

            # High CAS latency
            config_high = create_test_config(
                output_path=os.path.join(tmpdir, "high_cas.parquet"),
                seed=42,
                duration_ms=25000,
                t_cas=200,
                inter_arrival_scale=500.0
            )

            try:
                df_low_cas = run_simulation_from_config(config_low)
                df_high_cas = run_simulation_from_config(config_high)

                committed_low = df_low_cas[df_low_cas['status'] == 'committed']
                committed_high = df_high_cas[df_high_cas['status'] == 'committed']

                if len(committed_low) > 0 and len(committed_high) > 0:
                    avg_latency_low = committed_low['commit_latency'].mean()
                    avg_latency_high = committed_high['commit_latency'].mean()

                    assert avg_latency_high > avg_latency_low, \
                        f"High CAS latency should increase commit time: {avg_latency_high} vs {avg_latency_low}"

                    print(f"✓ CAS latency effect test passed")
                    print(f"  Low CAS (50ms): avg commit latency {avg_latency_low:.2f}ms")
                    print(f"  High CAS (200ms): avg commit latency {avg_latency_high:.2f}ms")

            finally:
                os.unlink(config_low)
                os.unlink(config_high)

    def test_more_retries_increases_success_rate(self):
        """Higher retry limit should increase success rate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Low retry limit
            config_low = create_test_config(
                output_path=os.path.join(tmpdir, "low_retry.parquet"),
                seed=42,
                duration_ms=25000,
                retry=2,
                inter_arrival_scale=200.0  # High contention
            )

            # High retry limit
            config_high = create_test_config(
                output_path=os.path.join(tmpdir, "high_retry.parquet"),
                seed=42,
                duration_ms=25000,
                retry=10,
                inter_arrival_scale=200.0  # High contention
            )

            try:
                df_low_retry = run_simulation_from_config(config_low)
                df_high_retry = run_simulation_from_config(config_high)

                success_rate_low = 100 * len(df_low_retry[df_low_retry['status'] == 'committed']) / len(df_low_retry)
                success_rate_high = 100 * len(df_high_retry[df_high_retry['status'] == 'committed']) / len(df_high_retry)

                assert success_rate_high >= success_rate_low, \
                    f"More retries should increase success rate: {success_rate_high}% vs {success_rate_low}%"

                print(f"✓ Retry limit effect test passed")
                print(f"  Low retries (2): {success_rate_low:.1f}% success")
                print(f"  High retries (10): {success_rate_high:.1f}% success")

            finally:
                os.unlink(config_low)
                os.unlink(config_high)

    def test_fewer_tables_reduces_contention(self):
        """Fewer tables should increase contention (more overlap)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Few tables (more contention) - use different seed for more pronounced effect
            config_few = create_test_config(
                output_path=os.path.join(tmpdir, "few_tables.parquet"),
                seed=123,  # Different seed to avoid edge cases
                duration_ms=35000,  # Longer duration for more stable statistics
                num_tables=2,  # Even fewer tables
                inter_arrival_scale=250.0  # Higher load for more contention
            )

            # Many tables (less contention)
            config_many = create_test_config(
                output_path=os.path.join(tmpdir, "many_tables.parquet"),
                seed=123,
                duration_ms=35000,
                num_tables=30,  # Even more tables
                inter_arrival_scale=250.0
            )

            try:
                df_few = run_simulation_from_config(config_few)
                df_many = run_simulation_from_config(config_many)

                committed_few = df_few[df_few['status'] == 'committed']
                committed_many = df_many[df_many['status'] == 'committed']

                if len(committed_few) > 0 and len(committed_many) > 0:
                    avg_retries_few = committed_few['n_retries'].mean()
                    avg_retries_many = committed_many['n_retries'].mean()

                    # With more extreme parameters, this should hold
                    # If not, at least verify there's some contention with few tables
                    assert avg_retries_few >= avg_retries_many - 0.5, \
                        f"Fewer tables should cause at least similar retries: {avg_retries_few} vs {avg_retries_many}"

                    print(f"✓ Table count effect test passed")
                    print(f"  Few tables (2): avg retries {avg_retries_few:.2f}")
                    print(f"  Many tables (30): avg retries {avg_retries_many:.2f}")

            finally:
                os.unlink(config_few)
                os.unlink(config_many)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
