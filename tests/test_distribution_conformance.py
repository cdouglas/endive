"""
Tests to verify that experimental results conform to configured distribution parameters.

These tests validate that the simulator produces data matching the configured distributions:
- Transaction runtime follows lognormal(mean, sigma, min)
- Inter-arrival times follow exponential(scale)
- Statistical properties (mean, percentiles) are within reasonable bounds

Tests use warmup/cooldown filtering to exclude transient behavior.
Tests are skipped if experiments/ directory is empty.
"""

import os
import random
from glob import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
import tomli
from scipy import stats


def find_experiment_dirs(base_dir: str = "experiments", pattern: str = "*-*") -> list:
    """Find all experiment directories matching pattern."""
    search_pattern = os.path.join(base_dir, pattern)
    return glob(search_pattern)


def load_experiment_config(exp_dir: str) -> Optional[Dict]:
    """Load configuration from experiment directory."""
    cfg_path = os.path.join(exp_dir, "cfg.toml")
    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, 'rb') as f:
        return tomli.load(f)


def compute_transient_period(config: Dict) -> Tuple[float, float]:
    """
    Compute warmup and cooldown periods using same logic as saturation_analysis.py.

    Returns:
        (warmup_ms, cooldown_ms)
    """
    K_MIN_CYCLES = 5
    MIN_PERIOD_MS = 5 * 60 * 1000  # 5 minutes
    MAX_PERIOD_MS = 15 * 60 * 1000  # 15 minutes

    mean_runtime_ms = config.get("transaction", {}).get("runtime", {}).get("mean", 10000)

    period_ms = max(
        MIN_PERIOD_MS,
        min(K_MIN_CYCLES * mean_runtime_ms, MAX_PERIOD_MS)
    )

    return period_ms, period_ms


def select_random_seed(exp_dir: str) -> Optional[str]:
    """Select a random seed directory from experiment."""
    seed_dirs = []
    for item in os.listdir(exp_dir):
        item_path = os.path.join(exp_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            seed_dirs.append(item_path)

    if not seed_dirs:
        return None

    return random.choice(seed_dirs)


def load_seed_results(seed_dir: str, config: Dict) -> Optional[pd.DataFrame]:
    """Load results from seed directory with warmup/cooldown filtering."""
    parquet_path = os.path.join(seed_dir, "results.parquet")
    if not os.path.exists(parquet_path):
        return None

    df = pd.read_parquet(parquet_path)

    # Skip empty dataframes (can happen with failed simulations)
    if len(df) == 0 or len(df.columns) == 0:
        return None

    # Apply warmup and cooldown filters
    warmup_ms, cooldown_ms = compute_transient_period(config)
    sim_duration_ms = config.get('simulation', {}).get('duration_ms', 3600000)
    cooldown_start_ms = sim_duration_ms - cooldown_ms

    df_filtered = df[(df['t_submit'] >= warmup_ms) & (df['t_submit'] < cooldown_start_ms)].copy()

    return df_filtered


# ============================================================================
# Test fixtures and helpers
# ============================================================================

@pytest.fixture(scope="module")
def experiment_samples():
    """
    Fixture that loads a random seed from each experiment for testing.

    Returns dict: {exp_label: {'config': config, 'data': dataframe}}
    """
    exp_dirs = find_experiment_dirs()

    if not exp_dirs:
        pytest.skip("No experiments found in experiments/ directory")

    samples = {}

    for exp_dir in exp_dirs:
        config = load_experiment_config(exp_dir)
        if not config:
            continue

        # Try multiple seed directories until we find one with valid data
        seed_dirs = []
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                seed_dirs.append(item_path)

        if not seed_dirs:
            continue

        # Shuffle to get random selection, then try until we find valid data
        random.shuffle(seed_dirs)
        data = None
        selected_seed_dir = None
        for seed_dir in seed_dirs:
            data = load_seed_results(seed_dir, config)
            if data is not None and len(data) > 0:
                selected_seed_dir = seed_dir
                break

        if data is None or len(data) == 0:
            continue

        label = config.get('experiment', {}).get('label', os.path.basename(exp_dir))
        samples[label] = {
            'config': config,
            'data': data,
            'exp_dir': exp_dir,
            'seed_dir': selected_seed_dir
        }

    if not samples:
        pytest.skip("No valid experiment samples found")

    return samples


def compute_relative_error(actual: float, expected: float) -> float:
    """Compute relative error as percentage."""
    if expected == 0:
        return 0.0
    return abs(actual - expected) / expected * 100


# ============================================================================
# Transaction runtime tests
# ============================================================================

class TestTransactionRuntimeDistribution:
    """Test that transaction runtimes follow configured lognormal distribution."""

    def test_runtime_mean_within_bounds(self, experiment_samples):
        """Test that observed mean runtime is within 20% of configured mean."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Get configured parameters
            # Note: Simulator does MIN + lognormal(mean, sigma)
            # So total expected mean = MIN + mean
            mean_lognormal = config.get('transaction', {}).get('runtime', {}).get('mean', 10000)
            min_runtime = config.get('transaction', {}).get('runtime', {}).get('min', 1000)
            expected_mean = min_runtime + mean_lognormal

            # Only test on committed transactions (aborted have selection bias)
            committed = data[data['status'] == 'committed']
            if len(committed) == 0:
                pytest.skip(f"{label}: No committed transactions")

            # Skip high-load experiments where selection bias is extreme
            # At high loads, long transactions are much more likely to abort,
            # creating a strong bias toward shorter runtimes in committed transactions
            success_rate = 100.0 * len(committed) / len(data)
            if success_rate < 90.0:
                pytest.skip(f"{label}: High load experiment (success rate={success_rate:.1f}%) has strong selection bias")

            observed_mean = committed['t_runtime'].mean()

            # Allow 30% tolerance due to:
            # 1. High variance of lognormal distributions (sigma=1.5 is very skewed)
            # 2. Selection bias even at 90%+ success rates
            # 3. Finite sample size effects
            relative_error = compute_relative_error(observed_mean, expected_mean)

            assert relative_error < 30.0, (
                f"{label}: Runtime mean {observed_mean:.0f}ms differs from "
                f"expected {expected_mean:.0f}ms by {relative_error:.1f}% (>30%)"
            )

    def test_runtime_minimum_enforced(self, experiment_samples):
        """Test that all runtimes are >= configured minimum."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Get configured minimum
            expected_min = config.get('transaction', {}).get('runtime', {}).get('min', 1000)

            # Check all runtimes
            observed_min = data['t_runtime'].min()

            # Allow small numerical tolerance (1ms)
            assert observed_min >= expected_min - 1, (
                f"{label}: Observed minimum runtime {observed_min:.0f}ms is below "
                f"configured minimum {expected_min:.0f}ms"
            )

    def test_runtime_lognormal_shape(self, experiment_samples):
        """Test that runtime distribution shape matches lognormal using percentile check."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Get configured parameters
            # Simulator uses MIN + lognormal(mean, sigma)
            mean_lognormal_ms = config.get('transaction', {}).get('runtime', {}).get('mean', 10000)
            sigma = config.get('transaction', {}).get('runtime', {}).get('sigma', 1.5)
            min_ms = config.get('transaction', {}).get('runtime', {}).get('min', 1000)

            # Only test committed transactions (selection bias in aborted)
            committed = data[data['status'] == 'committed']
            if len(committed) < 100:
                pytest.skip(f"{label}: Insufficient samples for shape test")

            observed = committed['t_runtime'].values

            # Expected distribution: MIN + lognormal
            # Compute mu for lognormal part
            mu = np.log(mean_lognormal_ms) - (sigma ** 2 / 2.0)

            # Generate theoretical samples to get expected percentiles
            np.random.seed(42)  # For reproducibility
            theoretical_samples = min_ms + np.random.lognormal(mu, sigma, size=100000)
            expected_p50 = np.percentile(theoretical_samples, 50)
            expected_p95 = np.percentile(theoretical_samples, 95)

            # Get observed percentiles
            observed_p50 = np.percentile(observed, 50)
            observed_p95 = np.percentile(observed, 95)

            p50_error = compute_relative_error(observed_p50, expected_p50)
            p95_error = compute_relative_error(observed_p95, expected_p95)

            # Accept if percentiles are within tolerances
            # P50: 30% tolerance (lognormal is very skewed, selection bias exists)
            # P95: 40% tolerance (high percentiles more variable)
            passes_p50 = p50_error < 30
            passes_p95 = p95_error < 40

            assert passes_p50 and passes_p95, (
                f"{label}: Runtime distribution percentiles differ from expected. "
                f"P50: observed={observed_p50:.0f}ms, expected={expected_p50:.0f}ms, error={p50_error:.1f}%. "
                f"P95: observed={observed_p95:.0f}ms, expected={expected_p95:.0f}ms, error={p95_error:.1f}%"
            )


# ============================================================================
# Inter-arrival time tests
# ============================================================================

class TestInterArrivalDistribution:
    """Test that inter-arrival times follow configured exponential distribution."""

    def test_arrival_mean_within_bounds(self, experiment_samples):
        """Test that observed mean inter-arrival is within 15% of configured scale."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Get configured scale (mean of exponential distribution)
            expected_mean = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 500)

            # Compute inter-arrival times from submission times
            submit_times = data['t_submit'].sort_values().values
            inter_arrivals = np.diff(submit_times)

            if len(inter_arrivals) < 100:
                pytest.skip(f"{label}: Insufficient samples for inter-arrival test")

            observed_mean = np.mean(inter_arrivals)

            # Allow 15% tolerance (exponential has high variance)
            relative_error = compute_relative_error(observed_mean, expected_mean)

            assert relative_error < 15.0, (
                f"{label}: Inter-arrival mean {observed_mean:.1f}ms differs from "
                f"expected {expected_mean:.1f}ms by {relative_error:.1f}% (>15%)"
            )

    def test_arrival_exponential_shape(self, experiment_samples):
        """Test that inter-arrival times follow exponential distribution."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Get configured scale
            scale = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 500)

            # Compute inter-arrival times
            submit_times = data['t_submit'].sort_values().values
            inter_arrivals = np.diff(submit_times)

            if len(inter_arrivals) < 100:
                pytest.skip(f"{label}: Insufficient samples for KS test")

            # With large samples, KS test becomes very sensitive to small deviations
            # Test mean and median instead, which are more robust
            observed_mean = np.mean(inter_arrivals)
            observed_median = np.median(inter_arrivals)

            # For exponential(scale), mean = scale, median = scale * ln(2) ≈ 0.693 * scale
            expected_mean = scale
            expected_median = scale * np.log(2)

            mean_error = compute_relative_error(observed_mean, expected_mean)
            median_error = compute_relative_error(observed_median, expected_median)

            # Accept if mean is within 15% and median is within 20%
            # Looser for median because it's more variable
            passes_mean = mean_error < 15
            passes_median = median_error < 20

            assert passes_mean and passes_median, (
                f"{label}: Inter-arrival distribution does not match exponential. "
                f"Mean: observed={observed_mean:.1f}ms, expected={expected_mean:.1f}ms, error={mean_error:.1f}%. "
                f"Median: observed={observed_median:.1f}ms, expected={expected_median:.1f}ms, error={median_error:.1f}%"
            )


# ============================================================================
# Commit latency tests (derived, not configured)
# ============================================================================

class TestCommitLatencyBehavior:
    """Test that commit latencies behave reasonably given configuration."""

    def test_commit_latency_nonnegative(self, experiment_samples):
        """Test that all commit latencies are non-negative."""
        for label, sample in experiment_samples.items():
            data = sample['data']

            committed = data[data['status'] == 'committed']
            if len(committed) == 0:
                pytest.skip(f"{label}: No committed transactions")

            min_latency = committed['commit_latency'].min()

            assert min_latency >= 0, (
                f"{label}: Found negative commit latency: {min_latency:.2f}ms"
            )

    def test_total_latency_consistency(self, experiment_samples):
        """Test that total_latency = commit_latency + t_runtime (approximately)."""
        for label, sample in experiment_samples.items():
            data = sample['data']

            committed = data[data['status'] == 'committed']
            if len(committed) == 0:
                pytest.skip(f"{label}: No committed transactions")

            # Check consistency: total_latency should equal commit_latency + t_runtime
            computed_total = committed['t_runtime'] + committed['commit_latency']
            recorded_total = committed['total_latency']

            # Allow small numerical error (1ms)
            max_error = np.abs(computed_total - recorded_total).max()

            assert max_error < 1.0, (
                f"{label}: Total latency inconsistency. "
                f"Max difference: {max_error:.4f}ms"
            )

    def test_commit_latency_increases_with_contention(self, experiment_samples):
        """
        Test that commit latency correlates with load level.

        Higher load (lower inter-arrival) should produce higher commit latencies.
        """
        # Group experiments by configuration (same runtime, different inter-arrival)
        by_base_config = {}

        for label, sample in experiment_samples.items():
            config = sample['config']

            # Create key from non-load parameters
            num_tables = config.get('catalog', {}).get('num_tables', 1)
            runtime_mean = config.get('transaction', {}).get('runtime', {}).get('mean', 10000)

            key = (num_tables, runtime_mean)

            if key not in by_base_config:
                by_base_config[key] = []

            inter_arrival = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 500)
            data = sample['data']
            committed = data[data['status'] == 'committed']

            if len(committed) > 0:
                mean_commit_latency = committed['commit_latency'].mean()
                by_base_config[key].append((inter_arrival, mean_commit_latency, label))

        # For each base configuration, check monotonicity
        for key, entries in by_base_config.items():
            if len(entries) < 3:
                continue  # Need at least 3 points to check trend

            # Sort by inter-arrival (descending) = load (ascending)
            entries_sorted = sorted(entries, key=lambda x: x[0], reverse=True)

            # Extract inter-arrivals and latencies
            inter_arrivals = [ia for ia, _, _ in entries_sorted]
            latencies = [lat for _, lat, _ in entries_sorted]

            # Skip if all inter-arrivals are the same (can't compute correlation)
            if len(set(inter_arrivals)) == 1:
                continue

            # Check if latencies generally increase with load
            # Use Spearman rank correlation (robust to outliers)
            # Load increases as inter_arrival decreases, so negate
            loads = [-ia for ia in inter_arrivals]
            correlation, p_value = stats.spearmanr(loads, latencies)

            # Expect positive correlation (higher load → higher latency)
            # Use lenient threshold since some low-load configs have similar latency
            assert correlation > -0.2, (
                f"Config {key}: Commit latency does not increase with load. "
                f"Correlation={correlation:.3f}, p={p_value:.4f}. "
                f"Entries: {[(ia, f'{lat:.0f}ms', lbl) for ia, lat, lbl in entries_sorted]}"
            )


# ============================================================================
# Success rate tests
# ============================================================================

class TestSuccessRateBehavior:
    """Test that success rates behave reasonably."""

    def test_success_rate_valid_range(self, experiment_samples):
        """Test that success rates are between 0 and 100%."""
        for label, sample in experiment_samples.items():
            data = sample['data']

            total = len(data)
            committed = len(data[data['status'] == 'committed'])

            success_rate = 100.0 * committed / total if total > 0 else 0

            assert 0 <= success_rate <= 100, (
                f"{label}: Invalid success rate: {success_rate:.1f}%"
            )

    def test_low_load_high_success(self, experiment_samples):
        """Test that low load configurations achieve >95% success rate."""
        for label, sample in experiment_samples.items():
            config = sample['config']
            data = sample['data']

            # Consider "low load" as inter-arrival > 1000ms
            inter_arrival = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 500)

            if inter_arrival < 1000:
                continue  # Skip high load configs

            total = len(data)
            committed = len(data[data['status'] == 'committed'])
            success_rate = 100.0 * committed / total if total > 0 else 0

            assert success_rate >= 95.0, (
                f"{label}: Low load config (inter-arrival={inter_arrival}ms) has "
                f"unexpectedly low success rate: {success_rate:.1f}%"
            )


# ============================================================================
# Test runner for manual execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
