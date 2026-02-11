"""
Tests for realistic latency modeling based on cloud provider measurements.

These tests validate:
1. Lognormal distribution generation
2. Configuration parsing for both legacy and new formats
3. Provider profiles (deferred to Phase 2)
4. Latency percentiles match expected values
"""

import numpy as np
import pytest
from scipy import stats

import endive.main as main


class TestLognormalDistribution:
    """Test lognormal latency generation."""

    def test_generate_latency_lognormal_positive(self):
        """Lognormal latency should always be positive."""
        main.MIN_LATENCY = 0.1
        mu = np.log(50)  # median = 50ms
        sigma = 0.5

        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(1000)]

        assert all(s >= main.MIN_LATENCY for s in samples)
        assert min(samples) >= main.MIN_LATENCY

    def test_generate_latency_lognormal_median(self):
        """Lognormal median should match exp(mu)."""
        main.MIN_LATENCY = 0.1
        target_median = 50.0
        mu = np.log(target_median)
        sigma = 0.3

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(10000)]
        observed_median = np.median(samples)

        # Allow 10% tolerance
        assert abs(observed_median - target_median) / target_median < 0.10

    def test_generate_latency_lognormal_shape(self):
        """Lognormal samples should pass K-S test against expected distribution."""
        main.MIN_LATENCY = 0.001  # Very low to not affect distribution
        mu = np.log(50)
        sigma = 0.4

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(5000)]

        # K-S test against lognormal distribution
        # Note: scipy uses different parameterization, scale=exp(mu)
        ks_stat, p_value = stats.kstest(samples, 'lognorm', args=(sigma, 0, np.exp(mu)))

        # With large sample, we expect p-value to be low but statistic should be small
        assert ks_stat < 0.05, f"K-S statistic {ks_stat} too high"

    def test_lognormal_heavier_tail_than_normal(self):
        """Lognormal should have heavier tail (higher p99/p50 ratio) than normal."""
        main.MIN_LATENCY = 0.1

        np.random.seed(42)

        # Lognormal with sigma=0.5
        mu = np.log(50)
        sigma = 0.5
        lognormal_samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(10000)]

        # Normal with similar mean
        normal_samples = [main.generate_latency(50, 10) for _ in range(10000)]

        lognormal_ratio = np.percentile(lognormal_samples, 99) / np.percentile(lognormal_samples, 50)
        normal_ratio = np.percentile(normal_samples, 99) / np.percentile(normal_samples, 50)

        assert lognormal_ratio > normal_ratio


class TestMuFromMedian:
    """Test mu parameter calculation from median."""

    def test_mu_from_median_basic(self):
        """mu should be ln(median)."""
        median = 50.0
        mu = main.lognormal_mu_from_median(median)
        assert abs(mu - np.log(50)) < 1e-10

    def test_mu_from_median_inverse(self):
        """exp(mu) should equal median."""
        for median in [10, 50, 100, 500]:
            mu = main.lognormal_mu_from_median(median)
            assert abs(np.exp(mu) - median) < 1e-10


class TestConvertMeanStddev:
    """Test conversion from mean/stddev to lognormal parameters."""

    def test_convert_preserves_approximate_mean(self):
        """Converted parameters should produce similar mean."""
        mean = 50.0
        stddev = 10.0

        mu, sigma = main.convert_mean_stddev_to_lognormal(mean, stddev)

        # Lognormal mean = exp(mu + sigma²/2)
        lognormal_mean = np.exp(mu + sigma ** 2 / 2)

        # Allow 20% tolerance due to approximation
        assert abs(lognormal_mean - mean) / mean < 0.20

    def test_convert_handles_zero_mean(self):
        """Should handle edge case of zero mean."""
        mu, sigma = main.convert_mean_stddev_to_lognormal(0, 10)
        assert sigma > 0  # Should return valid sigma
        assert np.isfinite(mu)

    def test_convert_clamps_sigma(self):
        """Sigma should be clamped to reasonable range."""
        # High CV case
        mu, sigma = main.convert_mean_stddev_to_lognormal(10, 100)
        assert 0.1 <= sigma <= 1.5


class TestParseLatencyConfig:
    """Test latency configuration parsing."""

    def test_parse_lognormal_format(self):
        """Should parse new lognormal format correctly."""
        config = {'median': 50.0, 'sigma': 0.4}
        result = main.parse_latency_config(config)

        assert result['distribution'] == 'lognormal'
        assert abs(result['mu'] - np.log(50)) < 1e-10
        assert result['sigma'] == 0.4

    def test_parse_lognormal_with_failure_multiplier(self):
        """Should preserve failure_multiplier."""
        config = {'median': 50.0, 'sigma': 0.4, 'failure_multiplier': 1.5}
        result = main.parse_latency_config(config)

        assert result['failure_multiplier'] == 1.5

    def test_parse_legacy_normal_format(self):
        """Should parse legacy mean/stddev format as normal."""
        config = {'mean': 50.0, 'stddev': 10.0}
        result = main.parse_latency_config(config)

        assert result['distribution'] == 'normal'
        assert result['mean'] == 50.0
        assert result['stddev'] == 10.0

    def test_parse_legacy_with_lognormal_override(self):
        """Should convert mean/stddev to lognormal when distribution specified."""
        config = {'mean': 50.0, 'stddev': 10.0, 'distribution': 'lognormal'}
        result = main.parse_latency_config(config)

        assert result['distribution'] == 'lognormal'
        assert 'mu' in result
        assert 'sigma' in result

    def test_parse_uses_defaults(self):
        """Should use defaults when config is empty."""
        defaults = {'mean': 100.0, 'stddev': 10.0, 'distribution': 'normal'}
        result = main.parse_latency_config({}, defaults)

        assert result == defaults


class TestGenerateLatencyFromConfig:
    """Test latency generation from config dict."""

    def test_generate_from_lognormal_config(self):
        """Should use lognormal distribution when configured."""
        main.MIN_LATENCY = 0.1
        config = {'mu': np.log(50), 'sigma': 0.3, 'distribution': 'lognormal'}

        np.random.seed(42)
        samples = [main.generate_latency_from_config(config) for _ in range(1000)]

        # Should have lognormal characteristics
        assert all(s >= main.MIN_LATENCY for s in samples)
        median = np.median(samples)
        assert abs(median - 50) / 50 < 0.15  # Within 15% of target median

    def test_generate_from_normal_config(self):
        """Should use normal distribution when configured."""
        main.MIN_LATENCY = 0.1
        config = {'mean': 50.0, 'stddev': 5.0, 'distribution': 'normal'}

        np.random.seed(42)
        samples = [main.generate_latency_from_config(config) for _ in range(1000)]

        mean = np.mean(samples)
        assert abs(mean - 50) / 50 < 0.10  # Within 10% of target mean


class TestProviderProfiles:
    """Test provider profile definitions."""

    def test_aws_profile_exists(self):
        """AWS profile should be defined."""
        assert 'aws' in main.PROVIDER_PROFILES
        aws = main.PROVIDER_PROFILES['aws']
        assert 'cas' in aws
        assert 'append' in aws
        assert 'manifest_list' in aws

    def test_aws_cas_parameters(self):
        """AWS CAS parameters should match measurements."""
        aws_cas = main.PROVIDER_PROFILES['aws']['cas']
        # AWS CAS: median=23ms, sigma=0.45
        assert aws_cas['median'] == 23
        assert aws_cas['sigma'] == 0.45

    def test_azure_profile_has_heavy_tails(self):
        """Azure should have higher sigma (heavier tails)."""
        aws_cas = main.PROVIDER_PROFILES['aws']['cas']
        azure_cas = main.PROVIDER_PROFILES['azure']['cas']

        assert azure_cas['sigma'] > aws_cas['sigma']

    def test_instant_profile_fast(self):
        """Instant profile should have very low latency."""
        instant = main.PROVIDER_PROFILES['instant']
        assert instant['cas']['median'] == 1
        assert instant['append']['median'] == 1


class TestBackwardCompatibility:
    """Test that existing configs work correctly."""

    def test_legacy_config_runs_simulation(self, tmp_path):
        """Legacy config with mean/stddev should run."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
max_parallel = 4
min_latency = 1
T_CAS.mean = 50
T_CAS.stddev = 5
T_METADATA_ROOT.read.mean = 10
T_METADATA_ROOT.read.stddev = 1
T_METADATA_ROOT.write.mean = 10
T_METADATA_ROOT.write.stddev = 1
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5
T_MANIFEST_LIST.write.mean = 60
T_MANIFEST_LIST.write.stddev = 6
T_MANIFEST_FILE.read.mean = 50
T_MANIFEST_FILE.read.stddev = 5
T_MANIFEST_FILE.write.mean = 60
T_MANIFEST_FILE.write.stddev = 6
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        # Should not raise
        main.configure_from_toml(str(config_file))

        # Verify T_CAS was loaded as normal distribution
        assert main.T_CAS['distribution'] == 'normal'
        assert main.T_CAS['mean'] == 50
        assert main.T_CAS['stddev'] == 5

    def test_new_lognormal_config_runs(self, tmp_path):
        """Config with median/sigma should work."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
max_parallel = 4
min_latency = 1
T_CAS.median = 23
T_CAS.sigma = 0.45
T_METADATA_ROOT.read.median = 10
T_METADATA_ROOT.read.sigma = 0.2
T_METADATA_ROOT.write.median = 10
T_METADATA_ROOT.write.sigma = 0.2
T_MANIFEST_LIST.read.median = 50
T_MANIFEST_LIST.read.sigma = 0.3
T_MANIFEST_LIST.write.median = 60
T_MANIFEST_LIST.write.sigma = 0.3
T_MANIFEST_FILE.read.median = 50
T_MANIFEST_FILE.read.sigma = 0.3
T_MANIFEST_FILE.write.median = 60
T_MANIFEST_FILE.write.sigma = 0.3
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Verify T_CAS was loaded as lognormal
        assert main.T_CAS['distribution'] == 'lognormal'
        assert abs(main.T_CAS['mu'] - np.log(23)) < 1e-10
        assert main.T_CAS['sigma'] == 0.45


class TestStorageCatalogSeparation:
    """Test storage vs catalog configuration separation."""

    def test_storage_provider_sets_manifest_latencies(self, tmp_path):
        """Storage provider should set manifest latencies."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Manifest latencies should come from AWS profile
        assert main.T_MANIFEST_LIST['read']['distribution'] == 'lognormal'
        # AWS manifest_list read: median=50, sigma=0.3
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 50) < 1

    def test_catalog_backend_storage_uses_storage_provider(self, tmp_path):
        """When backend=storage, catalog ops use storage provider."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
backend = "storage"

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # CAS should use AWS profile (median=23)
        assert main.T_CAS['distribution'] == 'lognormal'
        assert abs(np.exp(main.T_CAS['mu']) - 23) < 1

    def test_catalog_backend_service_uses_service_provider(self, tmp_path):
        """When backend=service, catalog ops use service provider."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
backend = "service"

[catalog.service]
provider = "instant"

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Storage uses AWS (manifest slow)
        assert main.T_MANIFEST_LIST['read']['distribution'] == 'lognormal'
        aws_manifest_median = np.exp(main.T_MANIFEST_LIST['read']['mu'])

        # Catalog uses instant (CAS fast)
        assert main.T_CAS['distribution'] == 'lognormal'
        instant_cas_median = np.exp(main.T_CAS['mu'])

        # AWS manifest should be much slower than instant CAS
        assert aws_manifest_median > instant_cas_median * 10

    def test_catalog_backend_service_explicit_latency(self, tmp_path):
        """Service backend can have explicit latency config."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
backend = "service"

[catalog.service]
T_CAS.median = 15.0
T_CAS.sigma = 0.3

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # CAS should use explicit service config
        assert main.T_CAS['distribution'] == 'lognormal'
        assert abs(np.exp(main.T_CAS['mu']) - 15) < 0.1
        assert main.T_CAS['sigma'] == 0.3

    def test_fifo_queue_backend(self, tmp_path):
        """FIFO queue backend uses queue config for append."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
mode = "append"
backend = "fifo_queue"

[catalog.fifo_queue]
append.median = 5.0
append.sigma = 0.2

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Append should use FIFO queue config (fast)
        assert main.T_APPEND['distribution'] == 'lognormal'
        queue_append_median = np.exp(main.T_APPEND['mu'])
        assert abs(queue_append_median - 5) < 0.1

        # Manifest still uses AWS storage
        aws_manifest_median = np.exp(main.T_MANIFEST_LIST['read']['mu'])
        assert aws_manifest_median > queue_append_median * 5

    def test_explicit_config_overrides_provider(self, tmp_path):
        """Explicit T_* config should override provider defaults."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
T_CAS.median = 100.0
T_CAS.sigma = 0.6
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # CAS should use explicit config, not AWS default
        assert main.T_CAS['distribution'] == 'lognormal'
        assert abs(np.exp(main.T_CAS['mu']) - 100) < 0.1
        assert main.T_CAS['sigma'] == 0.6


class TestConfigPrecedence:
    """Test configuration precedence rules (Phase 3)."""

    def test_no_provider_uses_explicit_config(self, tmp_path):
        """Without provider, explicit config is used directly."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
max_parallel = 4
min_latency = 1
T_CAS.median = 75.0
T_CAS.sigma = 0.5
T_MANIFEST_LIST.read.median = 100.0
T_MANIFEST_LIST.read.sigma = 0.4
T_MANIFEST_LIST.write.median = 120.0
T_MANIFEST_LIST.write.sigma = 0.4
T_MANIFEST_FILE.read.median = 100.0
T_MANIFEST_FILE.read.sigma = 0.4
T_MANIFEST_FILE.write.median = 120.0
T_MANIFEST_FILE.write.sigma = 0.4
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # All values should come from explicit config
        assert main.STORAGE_PROVIDER is None
        assert abs(np.exp(main.T_CAS['mu']) - 75) < 0.1
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 100) < 0.1

    def test_provider_plus_partial_override(self, tmp_path):
        """Provider sets defaults, explicit config overrides specific values."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
# Override just CAS, keep AWS defaults for manifests
T_CAS.median = 100.0
T_CAS.sigma = 0.6
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # CAS should use explicit config
        assert abs(np.exp(main.T_CAS['mu']) - 100) < 0.1
        assert main.T_CAS['sigma'] == 0.6

        # Manifests should use AWS profile defaults (median=50)
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 50) < 1

    def test_service_provider_overrides_storage_provider(self, tmp_path):
        """catalog.service.provider should override storage.provider for catalog ops."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
backend = "service"

[catalog.service]
provider = "azure"

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Storage uses AWS
        assert main.STORAGE_PROVIDER == "aws"
        # AWS manifest: median=50
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 50) < 1

        # Catalog service uses Azure
        # Azure CAS: median=75, sigma=0.80
        assert abs(np.exp(main.T_CAS['mu']) - 75) < 1
        assert abs(main.T_CAS['sigma'] - 0.80) < 0.01


class TestFailureLatencyMultiplier:
    """Test failure latency multiplier (Phase 4)."""

    def test_failure_multiplier_applied(self):
        """Failure latency should be multiplied when success=False."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = False

        # Set up T_CAS with known multiplier
        main.T_CAS = {
            'mu': np.log(50),
            'sigma': 0.3,
            'distribution': 'lognormal',
            'failure_multiplier': 2.0
        }

        np.random.seed(42)
        success_samples = [main.get_cas_latency(success=True) for _ in range(1000)]
        np.random.seed(42)  # Same seed
        failure_samples = [main.get_cas_latency(success=False) for _ in range(1000)]

        success_mean = np.mean(success_samples)
        failure_mean = np.mean(failure_samples)

        # Failure should be ~2x success
        assert abs(failure_mean / success_mean - 2.0) < 0.1

    def test_no_multiplier_default(self):
        """Without failure_multiplier, success and failure should be same."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = False

        main.T_CAS = {
            'mu': np.log(50),
            'sigma': 0.3,
            'distribution': 'lognormal'
            # No failure_multiplier
        }

        np.random.seed(42)
        success_samples = [main.get_cas_latency(success=True) for _ in range(1000)]
        np.random.seed(42)
        failure_samples = [main.get_cas_latency(success=False) for _ in range(1000)]

        # Should be identical
        assert success_samples == failure_samples

    def test_append_failure_multiplier(self):
        """Append operations should also support failure multiplier."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = False

        # Azure-like: append failures are 34x slower!
        main.T_APPEND = {
            'mu': np.log(77),
            'sigma': 0.28,
            'distribution': 'lognormal',
            'failure_multiplier': 34.3
        }

        np.random.seed(42)
        success_samples = [main.get_append_latency(success=True) for _ in range(1000)]
        np.random.seed(42)
        failure_samples = [main.get_append_latency(success=False) for _ in range(1000)]

        success_mean = np.mean(success_samples)
        failure_mean = np.mean(failure_samples)

        # Failure should be ~34x success
        assert abs(failure_mean / success_mean - 34.3) < 1.0


class TestContentionScaling:
    """Test contention-based latency scaling (Phase 4)."""

    def test_contention_tracker_basic(self):
        """ContentionTracker should count operations."""
        tracker = main.ContentionTracker()

        assert tracker.active_cas == 0
        tracker.enter_cas()
        assert tracker.active_cas == 1
        tracker.enter_cas()
        assert tracker.active_cas == 2
        tracker.exit_cas()
        assert tracker.active_cas == 1
        tracker.exit_cas()
        assert tracker.active_cas == 0

    def test_contention_factor_calculation(self):
        """Contention factor should scale linearly from 1.0 to scaling value."""
        tracker = main.ContentionTracker()
        main.CONTENTION_SCALING_ENABLED = True
        main.CATALOG_CONTENTION_SCALING = {'cas': 1.4, 'append': 1.8}

        # At 1 concurrent: factor = 1.0
        tracker.active_cas = 1
        assert tracker.get_contention_factor("cas") == 1.0

        # At 16 concurrent: factor = 1.4
        tracker.active_cas = 16
        factor_16 = tracker.get_contention_factor("cas")
        assert abs(factor_16 - 1.4) < 0.01

        # At 8 concurrent: factor = 1.0 + (1.4-1.0) * 7/15 ≈ 1.187
        tracker.active_cas = 8
        factor_8 = tracker.get_contention_factor("cas")
        expected = 1.0 + 0.4 * 7 / 15
        assert abs(factor_8 - expected) < 0.01

    def test_contention_scaling_disabled(self):
        """When disabled, contention factor should be 1.0."""
        tracker = main.ContentionTracker()
        main.CONTENTION_SCALING_ENABLED = False
        main.CATALOG_CONTENTION_SCALING = {'cas': 1.4}

        tracker.active_cas = 16
        assert tracker.get_contention_factor("cas") == 1.0

    def test_contention_scaling_applied_to_latency(self):
        """Latency should increase with contention when enabled."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = True
        main.CATALOG_CONTENTION_SCALING = {'cas': 2.0}  # 2x at 16 threads
        main.CONTENTION_TRACKER.reset()

        main.T_CAS = {
            'mu': np.log(50),
            'sigma': 0.3,
            'distribution': 'lognormal'
        }

        # Low contention
        main.CONTENTION_TRACKER.active_cas = 1
        np.random.seed(42)
        low_contention = [main.get_cas_latency() for _ in range(1000)]

        # High contention
        main.CONTENTION_TRACKER.active_cas = 16
        np.random.seed(42)
        high_contention = [main.get_cas_latency() for _ in range(1000)]

        low_mean = np.mean(low_contention)
        high_mean = np.mean(high_contention)

        # High contention should be ~2x low contention
        assert abs(high_mean / low_mean - 2.0) < 0.1

        # Cleanup
        main.CONTENTION_SCALING_ENABLED = False
        main.CONTENTION_TRACKER.reset()

    def test_contention_scaling_auto_enabled_with_provider(self, tmp_path):
        """Contention scaling should auto-enable when provider is specified."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        assert main.CONTENTION_SCALING_ENABLED == True
        assert main.CATALOG_CONTENTION_SCALING.get('cas') == 1.4
        assert main.CATALOG_CONTENTION_SCALING.get('append') == 1.8

    def test_contention_scaling_can_be_disabled(self, tmp_path):
        """Contention scaling can be explicitly disabled."""
        config_content = """
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500

[storage]
provider = "aws"
contention_scaling_enabled = false
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        assert main.CONTENTION_SCALING_ENABLED == False


class TestLatencyPercentiles:
    """Test that generated latencies match expected percentiles."""

    def test_aws_cas_percentiles(self):
        """Generated AWS CAS latencies should match measured percentiles."""
        main.MIN_LATENCY = 0.1
        # AWS CAS: median=23ms, p95=65ms, p99=78ms
        mu = np.log(23)
        sigma = 0.45

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p99 = np.percentile(samples, 99)

        # Allow 20% tolerance
        assert abs(p50 - 23) / 23 < 0.20, f"p50={p50}, expected ~23"
        assert abs(p95 - 65) / 65 < 0.30, f"p95={p95}, expected ~65"
        assert abs(p99 - 78) / 78 < 0.30, f"p99={p99}, expected ~78"

    def test_aws_append_percentiles(self):
        """Generated AWS append latencies should match measured percentiles."""
        main.MIN_LATENCY = 0.1
        # AWS append: median=20ms, p95=28ms, p99=48ms
        mu = np.log(20)
        sigma = 0.25

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)

        assert abs(p50 - 20) / 20 < 0.20, f"p50={p50}, expected ~20"
        assert abs(p95 - 28) / 28 < 0.30, f"p95={p95}, expected ~28"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
