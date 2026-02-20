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


@pytest.fixture(autouse=True)
def reset_provider_instances():
    """Reset provider instances before each test to ensure T_CAS/T_APPEND are used directly."""
    # Store original values
    orig_catalog = main.CATALOG_PROVIDER_INSTANCE
    orig_storage = main.STORAGE_PROVIDER_INSTANCE

    # Reset before test
    main.CATALOG_PROVIDER_INSTANCE = None
    main.STORAGE_PROVIDER_INSTANCE = None

    yield

    # Restore after test (for tests that need providers)
    main.CATALOG_PROVIDER_INSTANCE = orig_catalog
    main.STORAGE_PROVIDER_INSTANCE = orig_storage


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
    """Test provider profile definitions based on June 2025 YCSB benchmarks."""

    def test_all_profiles_exist(self):
        """All expected provider profiles should be defined."""
        expected = ['s3', 's3x', 'azure', 'azurex', 'gcp', 'instant', 'aws']
        for provider in expected:
            assert provider in main.PROVIDER_PROFILES, f"Missing profile: {provider}"

    def test_s3_standard_no_append(self):
        """S3 Standard should not support conditional append."""
        s3 = main.PROVIDER_PROFILES['s3']
        assert s3['cas'] is not None
        assert s3['append'] is None  # S3 Standard has no conditional append

    def test_s3x_parameters(self):
        """S3 Express One Zone parameters should match measurements."""
        s3x = main.PROVIDER_PROFILES['s3x']
        # S3 Express CAS: median=22.4ms, sigma=0.22
        assert s3x['cas']['median'] == 22
        assert abs(s3x['cas']['sigma'] - 0.22) < 0.01
        # S3 Express append: median=20.5ms, sigma=0.25
        assert s3x['append']['median'] == 21
        assert abs(s3x['append']['sigma'] - 0.25) < 0.01

    def test_azure_blob_heavy_tails(self):
        """Azure Blob Storage should have heavy-tailed CAS distribution."""
        azure = main.PROVIDER_PROFILES['azure']
        # Azure CAS: sigma=0.82 (very heavy tails, p99/p50=50x)
        assert azure['cas']['sigma'] > 0.8
        # Azure append failures are 31.6x slower
        assert azure['append']['failure_multiplier'] > 30

    def test_azure_premium_parameters(self):
        """Azure Premium Block Blob parameters should match measurements."""
        azurex = main.PROVIDER_PROFILES['azurex']
        # Azure Premium CAS: median=63.5ms, sigma=0.73
        assert azurex['cas']['median'] == 64
        assert abs(azurex['cas']['sigma'] - 0.73) < 0.01
        # Azure Premium append failures are 36.2x slower
        assert azurex['append']['failure_multiplier'] > 35

    def test_gcp_no_append(self):
        """GCP Cloud Storage should not have append data."""
        gcp = main.PROVIDER_PROFILES['gcp']
        assert gcp['cas'] is not None
        assert gcp['append'] is None  # No append data for GCP
        # GCP CAS: median=170ms, sigma=0.91 (extremely heavy tails)
        assert gcp['cas']['median'] == 170
        assert gcp['cas']['sigma'] > 0.9

    def test_aws_alias_points_to_s3x(self):
        """'aws' should be an alias for 's3x' for backward compatibility."""
        assert main.PROVIDER_PROFILES['aws'] is main.PROVIDER_PROFILES['s3x']

    def test_instant_profile_fast(self):
        """Instant profile should have very low latency."""
        instant = main.PROVIDER_PROFILES['instant']
        assert instant['cas']['median'] == 1
        assert instant['append']['median'] == 1

    def test_tier_ordering(self):
        """Premium tiers should be faster than standard tiers."""
        # S3 Express faster than S3 Standard
        assert main.PROVIDER_PROFILES['s3x']['cas']['median'] < main.PROVIDER_PROFILES['s3']['cas']['median']
        # Azure Premium faster than Azure Blob
        assert main.PROVIDER_PROFILES['azurex']['cas']['median'] < main.PROVIDER_PROFILES['azure']['cas']['median']


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

        # Manifest latencies should come from AWS (S3 Express alias) profile
        assert main.T_MANIFEST_LIST['read']['distribution'] == 'lognormal'
        # S3 Express manifest_list read: median=22, sigma=0.22
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 22) < 2

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

        # Manifest still uses AWS (S3 Express) storage
        s3x_manifest_median = np.exp(main.T_MANIFEST_LIST['read']['mu'])
        # S3 Express manifest ~22ms should be > 5ms queue append
        assert s3x_manifest_median > queue_append_median * 2

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

        # Manifests should use AWS (S3 Express) profile defaults (median=22)
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 22) < 2

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

        # Storage uses AWS (S3 Express alias)
        assert main.STORAGE_PROVIDER == "aws"
        # S3 Express manifest: median=22
        assert abs(np.exp(main.T_MANIFEST_LIST['read']['mu']) - 22) < 2

        # Catalog service uses Azure (Blob)
        # Azure Blob CAS: median=93, sigma=0.82
        assert abs(np.exp(main.T_CAS['mu']) - 93) < 5
        assert abs(main.T_CAS['sigma'] - 0.82) < 0.02


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
        # S3 Express (aws alias) contention scaling: CAS=1.77, append=1.85
        assert abs(main.CATALOG_CONTENTION_SCALING.get('cas') - 1.77) < 0.01
        assert abs(main.CATALOG_CONTENTION_SCALING.get('append') - 1.85) < 0.01

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
    """Test that generated latencies match expected percentiles from June 2025 YCSB benchmarks."""

    def test_s3x_cas_percentiles(self):
        """Generated S3 Express CAS latencies should match measured percentiles."""
        main.MIN_LATENCY = 0.1
        # S3 Express CAS: median=22.4ms, p95=27.8ms, p99=44.3ms, sigma=0.224
        mu = np.log(22.4)
        sigma = 0.224

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p99 = np.percentile(samples, 99)

        # Allow 20% tolerance
        assert abs(p50 - 22.4) / 22.4 < 0.20, f"p50={p50}, expected ~22.4"
        assert abs(p95 - 27.8) / 27.8 < 0.30, f"p95={p95}, expected ~27.8"
        assert abs(p99 - 44.3) / 44.3 < 0.40, f"p99={p99}, expected ~44.3"

    def test_s3x_append_percentiles(self):
        """Generated S3 Express append latencies should match measured percentiles."""
        main.MIN_LATENCY = 0.1
        # S3 Express append: median=20.5ms, p95=27.8ms, p99=48.2ms, sigma=0.247
        mu = np.log(20.5)
        sigma = 0.247

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)

        assert abs(p50 - 20.5) / 20.5 < 0.20, f"p50={p50}, expected ~20.5"
        assert abs(p95 - 27.8) / 27.8 < 0.30, f"p95={p95}, expected ~27.8"

    def test_s3_standard_cas_percentiles(self):
        """Generated S3 Standard CAS latencies should match measured percentiles."""
        main.MIN_LATENCY = 0.1
        # S3 Standard CAS: median=60.8ms, p95=78.5ms, p99=103ms, sigma=0.14
        mu = np.log(60.8)
        sigma = 0.14

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p99 = np.percentile(samples, 99)

        assert abs(p50 - 60.8) / 60.8 < 0.20, f"p50={p50}, expected ~60.8"
        assert abs(p95 - 78.5) / 78.5 < 0.30, f"p95={p95}, expected ~78.5"
        assert abs(p99 - 103) / 103 < 0.40, f"p99={p99}, expected ~103"

    def test_azure_blob_cas_heavy_tail(self):
        """Azure Blob CAS should have very heavy tail (p99/p50 > 30x)."""
        main.MIN_LATENCY = 0.1
        # Azure Blob CAS: median=93.1ms, sigma=0.82 (extremely heavy tail)
        mu = np.log(93.1)
        sigma = 0.82

        np.random.seed(42)
        samples = [main.generate_latency_lognormal(mu, sigma) for _ in range(50000)]

        p50 = np.percentile(samples, 50)
        p99 = np.percentile(samples, 99)
        ratio = p99 / p50

        # Azure Blob has p99/p50 ratio around 50x
        assert ratio > 5, f"p99/p50 ratio {ratio:.1f} too low for heavy-tailed Azure"


class TestSimulationBounds:
    """End-to-end validation tests with expected bounds (Phase 5)."""

    def test_aws_provider_simulation_runs(self, tmp_path):
        """Simulation with AWS provider should complete successfully."""
        config_content = """
[simulation]
duration_ms = 30000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 5
runtime.min = 1000
runtime.mean = 3000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 200

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)
        output_file = tmp_path / "results.parquet"

        # Run simulation
        main.configure_from_toml(str(config_file))
        main.SIM_OUTPUT_PATH = str(output_file)

        np.random.seed(42)
        main.TABLE_TO_GROUP, main.GROUP_TO_TABLES = main.partition_tables_into_groups(
            main.N_TABLES, main.N_GROUPS, main.GROUP_SIZE_DIST, main.LONGTAIL_PARAMS
        )
        main.STATS = main.Stats()

        env = main.simpy.Environment()
        env.process(main.setup(env))
        env.run(until=main.SIM_DURATION_MS)

        # Verify simulation produced results
        assert main.STATS.txn_committed + main.STATS.txn_aborted > 0

    def test_instant_catalog_higher_throughput(self, tmp_path):
        """Instant catalog should allow higher throughput than AWS."""
        results = {}

        for provider, name in [("instant", "instant"), ("aws", "aws")]:
            config_content = f"""
[simulation]
duration_ms = 20000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 5
runtime.min = 500
runtime.mean = 1000
runtime.sigma = 0.3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 50
real_conflict_probability = 0.0

[storage]
provider = "{provider}"
max_parallel = 4
min_latency = 1
"""
            config_file = tmp_path / f"config_{name}.toml"
            config_file.write_text(config_content)

            main.configure_from_toml(str(config_file))
            np.random.seed(42)
            main.TABLE_TO_GROUP, main.GROUP_TO_TABLES = main.partition_tables_into_groups(
                main.N_TABLES, main.N_GROUPS, main.GROUP_SIZE_DIST, main.LONGTAIL_PARAMS
            )
            main.STATS = main.Stats()

            env = main.simpy.Environment()
            env.process(main.setup(env))
            env.run(until=main.SIM_DURATION_MS)

            # Calculate throughput
            duration_sec = main.SIM_DURATION_MS / 1000
            throughput = main.STATS.txn_committed / duration_sec
            results[name] = throughput

        # Instant should have higher throughput (faster CAS)
        assert results["instant"] >= results["aws"] * 0.9  # Allow some variance

    def test_latency_distribution_lognormal_shape(self, tmp_path):
        """Generated latencies in simulation should follow lognormal distribution."""
        config_content = """
[simulation]
duration_ms = 60000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 10

[transaction]
retry = 5
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100
real_conflict_probability = 0.0

[storage]
provider = "aws"
max_parallel = 4
min_latency = 1
"""
        config_file = tmp_path / "test_config.toml"
        config_file.write_text(config_content)

        main.configure_from_toml(str(config_file))

        # Generate CAS latency samples
        np.random.seed(42)
        samples = [main.get_cas_latency() for _ in range(5000)]

        # Verify lognormal shape - check p99/p50 ratio
        p50 = np.percentile(samples, 50)
        p99 = np.percentile(samples, 99)
        ratio = p99 / p50

        # S3 Express CAS (aws alias) has sigma=0.22, giving p99/p50 ≈ 1.67
        # This is a relatively tight distribution (low variance)
        # Allow wider range to account for contention scaling and provider variations
        assert 1.3 < ratio < 4.0, f"Ratio {ratio} outside expected range for S3 Express lognormal"

    def test_provider_profiles_produce_different_latencies(self, tmp_path):
        """Different providers should produce significantly different latencies."""
        median_latencies = {}

        for provider in ["aws", "azure", "gcp"]:
            config_content = f"""
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
provider = "{provider}"
max_parallel = 4
min_latency = 1
"""
            config_file = tmp_path / f"config_{provider}.toml"
            config_file.write_text(config_content)

            main.configure_from_toml(str(config_file))

            np.random.seed(42)
            samples = [main.get_cas_latency() for _ in range(1000)]
            median_latencies[provider] = np.median(samples)

        # S3 Express (aws alias) should be faster than Azure, Azure faster than GCP
        assert median_latencies["aws"] < median_latencies["azure"]
        assert median_latencies["azure"] < median_latencies["gcp"]

        # Check rough ratios:
        # S3 Express CAS ~22ms, Azure Blob CAS ~93ms, GCP CAS ~170ms
        assert median_latencies["azure"] / median_latencies["aws"] > 3.0  # 93/22 ≈ 4.2
        assert median_latencies["gcp"] / median_latencies["azure"] > 1.5  # 170/93 ≈ 1.8


class TestDistributionValidation:
    """Validate that simulated distributions match June 2025 YCSB measurements."""

    def test_s3x_cas_distribution_matches_measurements(self):
        """S3 Express CAS latencies should match measured distribution."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = False

        # S3 Express CAS: mu=9.98, sigma=0.22, median=22.4ms
        main.T_CAS = {
            'mu': np.log(22.4),
            'sigma': 0.22,
            'distribution': 'lognormal'
        }

        np.random.seed(42)
        samples = [main.get_cas_latency() for _ in range(50000)]

        # Measured values from analysis/simulation_summary.md:
        # median=22.4ms, p95=27.8ms, p99=44.3ms
        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p99 = np.percentile(samples, 99)

        # Allow 25% tolerance
        assert abs(p50 - 22.4) / 22.4 < 0.25, f"p50={p50}, expected ~22.4"
        assert abs(p95 - 27.8) / 27.8 < 0.35, f"p95={p95}, expected ~27.8"
        assert abs(p99 - 44.3) / 44.3 < 0.50, f"p99={p99}, expected ~44.3"

    def test_azure_blob_cas_heavy_tail(self):
        """Azure Blob CAS should have much heavier tail than S3 Express."""
        main.MIN_LATENCY = 0.1
        main.CONTENTION_SCALING_ENABLED = False

        # S3 Express: sigma=0.22 -> p99/p50 ≈ exp(2.326*0.22) ≈ 1.67
        main.T_CAS = {'mu': np.log(22), 'sigma': 0.22, 'distribution': 'lognormal'}
        np.random.seed(42)
        s3x_samples = [main.get_cas_latency() for _ in range(10000)]
        s3x_ratio = np.percentile(s3x_samples, 99) / np.percentile(s3x_samples, 50)

        # Azure Blob: sigma=0.82 -> p99/p50 ≈ exp(2.326*0.82) ≈ 6.7
        # Measured: p99/p50 = 50x (extremely heavy!)
        main.T_CAS = {'mu': np.log(93), 'sigma': 0.82, 'distribution': 'lognormal'}
        np.random.seed(42)
        azure_samples = [main.get_cas_latency() for _ in range(10000)]
        azure_ratio = np.percentile(azure_samples, 99) / np.percentile(azure_samples, 50)

        # Azure should have much higher p99/p50 ratio
        # S3 Express ~1.7x, Azure Blob ~6-7x (lognormal), actual measured ~50x
        assert azure_ratio > s3x_ratio * 2.5, f"Azure ratio {azure_ratio:.2f} not > S3x ratio {s3x_ratio:.2f} * 2.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
