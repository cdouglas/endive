"""
Tests for storage provider configuration integration.

These tests validate:
1. Provider profile loading from config files
2. Experiment configs correctly use providers
3. Provider-specific latency settings
4. Integration between storage and catalog configurations
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

import endive.main as main
from endive.capstats import Stats


def create_minimal_config(provider: str = None, extra_storage: str = "",
                          extra_catalog: str = "", extra_transaction: str = "") -> str:
    """Create a minimal valid config for testing."""
    provider_line = f'provider = "{provider}"' if provider else ""
    return f"""
[simulation]
duration_ms = 10000
output_path = "results.parquet"
seed = 42

[catalog]
num_tables = 1
{extra_catalog}

[transaction]
retry = 3
runtime.min = 1000
runtime.mean = 5000
runtime.sigma = 0.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 500
{extra_transaction}

[storage]
{provider_line}
max_parallel = 4
min_latency = 1
{extra_storage}
"""


def write_temp_config(content: str) -> str:
    """Write config to temp file and return path."""
    fd, path = tempfile.mkstemp(suffix='.toml')
    os.write(fd, content.encode())
    os.close(fd)
    return path


class TestProviderConfigLoading:
    """Test that provider configs are correctly loaded."""

    def test_s3_provider_loads(self, tmp_path):
        """S3 provider config should load correctly."""
        config_path = write_temp_config(create_minimal_config(provider="s3"))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            # S3 Standard CAS median is 61ms
            assert main.T_CAS['distribution'] == 'lognormal'
            expected_median = 61
            actual_median = np.exp(main.T_CAS['mu'])
            assert abs(actual_median - expected_median) < 2
        finally:
            os.unlink(config_path)

    def test_s3x_provider_loads(self, tmp_path):
        """S3 Express provider config should load correctly."""
        config_path = write_temp_config(create_minimal_config(provider="s3x"))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            # S3 Express CAS median is 22ms
            assert main.T_CAS['distribution'] == 'lognormal'
            expected_median = 22
            actual_median = np.exp(main.T_CAS['mu'])
            assert abs(actual_median - expected_median) < 2
        finally:
            os.unlink(config_path)

    def test_azure_provider_loads(self, tmp_path):
        """Azure provider config should load correctly."""
        config_path = write_temp_config(create_minimal_config(provider="azure"))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            # Azure CAS median is 93ms
            assert main.T_CAS['distribution'] == 'lognormal'
            expected_median = 93
            actual_median = np.exp(main.T_CAS['mu'])
            assert abs(actual_median - expected_median) < 5
        finally:
            os.unlink(config_path)

    def test_provider_sets_manifest_latencies(self, tmp_path):
        """Provider should also set manifest list/file latencies."""
        config_path = write_temp_config(create_minimal_config(provider="s3x"))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            # Check manifest list latencies are set
            assert 'read' in main.T_MANIFEST_LIST
            assert 'write' in main.T_MANIFEST_LIST
            assert main.T_MANIFEST_LIST['read']['distribution'] == 'lognormal'
        finally:
            os.unlink(config_path)


class TestExperimentConfigProviders:
    """Test that experiment configs correctly specify providers."""

    @pytest.fixture
    def experiment_configs_dir(self):
        """Return path to experiment_configs directory."""
        return Path(__file__).parent.parent / "experiment_configs"

    def test_baseline_s3_uses_s3_provider(self, experiment_configs_dir):
        """baseline_s3.toml should use s3 provider."""
        import tomli

        config_path = experiment_configs_dir / "baseline_s3.toml"
        if not config_path.exists():
            pytest.skip("Config not found")

        with open(config_path, "rb") as f:
            data = tomli.load(f)

        assert data["storage"]["provider"] == "s3"

    def test_baseline_s3x_uses_s3x_provider(self, experiment_configs_dir):
        """baseline_s3x.toml should use s3x provider."""
        import tomli

        config_path = experiment_configs_dir / "baseline_s3x.toml"
        if not config_path.exists():
            pytest.skip("Config not found")

        with open(config_path, "rb") as f:
            data = tomli.load(f)

        assert data["storage"]["provider"] == "s3x"

    def test_baseline_azure_uses_azure_provider(self, experiment_configs_dir):
        """baseline_azure.toml should use azure provider."""
        import tomli

        config_path = experiment_configs_dir / "baseline_azure.toml"
        if not config_path.exists():
            pytest.skip("Config not found")

        with open(config_path, "rb") as f:
            data = tomli.load(f)

        assert data["storage"]["provider"] == "azure"

    def test_all_optimization_configs_have_providers(self, experiment_configs_dir):
        """All optimization configs should have a provider specified."""
        import tomli

        optimization_patterns = [
            "baseline_*.toml",
            "metadata_*.toml",
            "ml_append_*.toml",
            "combined_*.toml",
        ]

        from glob import glob

        for pattern in optimization_patterns:
            for config_path in glob(str(experiment_configs_dir / pattern)):
                if "archive" in config_path:
                    continue

                with open(config_path, "rb") as f:
                    data = tomli.load(f)

                assert "provider" in data.get("storage", {}), \
                    f"Missing provider in {config_path}"


class TestProviderLatencyDifferences:
    """Test that different providers produce different latencies."""

    def test_s3_slower_than_s3x(self):
        """S3 Standard should have higher latency than S3 Express."""
        s3 = main.PROVIDER_PROFILES['s3']
        s3x = main.PROVIDER_PROFILES['s3x']

        assert s3['cas']['median'] > s3x['cas']['median']

    def test_azure_slower_than_azure_premium(self):
        """Azure Blob should have higher latency than Azure Premium."""
        azure = main.PROVIDER_PROFILES['azure']
        azurex = main.PROVIDER_PROFILES['azurex']

        assert azure['cas']['median'] > azurex['cas']['median']

    def test_gcp_slowest_cas(self):
        """GCP should have the slowest CAS latency."""
        gcp = main.PROVIDER_PROFILES['gcp']

        for provider_name in ['s3', 's3x', 'azure', 'azurex']:
            provider = main.PROVIDER_PROFILES[provider_name]
            assert gcp['cas']['median'] > provider['cas']['median']


class TestMetadataInliningConfig:
    """Test table_metadata_inlined configuration."""

    def test_inlined_true_disables_metadata_io(self, tmp_path):
        """When inlined=true, table metadata I/O should be disabled."""
        config_path = write_temp_config(create_minimal_config(
            provider="s3x",
            extra_catalog="table_metadata_inlined = true"
        ))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            assert main.TABLE_METADATA_INLINED == True
        finally:
            os.unlink(config_path)

    def test_inlined_false_enables_metadata_io(self, tmp_path):
        """When inlined=false, table metadata I/O should be enabled."""
        config_path = write_temp_config(create_minimal_config(
            provider="s3x",
            extra_catalog="table_metadata_inlined = false"
        ))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            assert main.TABLE_METADATA_INLINED == False
        finally:
            os.unlink(config_path)


class TestManifestListModeConfig:
    """Test manifest_list_mode configuration."""

    def test_rewrite_mode(self, tmp_path):
        """Rewrite mode should be default/traditional Iceberg."""
        config_path = write_temp_config(create_minimal_config(
            provider="s3x",
            extra_transaction='manifest_list_mode = "rewrite"'
        ))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            assert main.MANIFEST_LIST_MODE == "rewrite"
        finally:
            os.unlink(config_path)

    def test_append_mode(self, tmp_path):
        """Append mode enables ML+ optimization."""
        config_path = write_temp_config(create_minimal_config(
            provider="s3x",
            extra_transaction='manifest_list_mode = "append"'
        ))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            assert main.MANIFEST_LIST_MODE == "append"
        finally:
            os.unlink(config_path)


class TestSizeDependentLatency:
    """Test size-dependent latency model for PUT operations."""

    def test_larger_size_higher_latency(self):
        """Larger files should have higher PUT latency on average."""
        # Set up T_PUT dict
        main.T_PUT = {
            'base_latency_ms': 30.0,
            'latency_per_mib_ms': 20.0,
            'sigma': 0.3,
        }
        main.MIN_LATENCY = 1.0

        # Use multiple samples to account for randomness
        np.random.seed(42)
        latencies_1mib = [main.get_put_latency(1 * 1024 * 1024) for _ in range(100)]
        latencies_10mib = [main.get_put_latency(10 * 1024 * 1024) for _ in range(100)]

        # On average, 10MiB should have higher latency than 1MiB
        assert np.median(latencies_10mib) > np.median(latencies_1mib)

    def test_put_latency_median_scales_with_size(self):
        """PUT latency median should scale approximately linearly with size."""
        main.T_PUT = {
            'base_latency_ms': 30.0,
            'latency_per_mib_ms': 20.0,
            'sigma': 0.3,
        }
        main.MIN_LATENCY = 1.0

        np.random.seed(42)
        latencies_5mib = [main.get_put_latency(5 * 1024 * 1024) for _ in range(1000)]

        # Expected deterministic value: 30 + 5 * 20 = 130ms (median of lognormal)
        expected_median = 30.0 + (5.0 * 20.0)
        observed_median = np.median(latencies_5mib)

        # Allow 10% tolerance due to sampling variance
        assert abs(observed_median - expected_median) / expected_median < 0.10

    def test_put_latency_respects_min_latency(self):
        """PUT latency should never be below MIN_LATENCY."""
        main.T_PUT = {
            'base_latency_ms': 0.5,
            'latency_per_mib_ms': 0.1,
            'sigma': 0.3,
        }
        main.MIN_LATENCY = 10.0

        np.random.seed(42)
        latencies = [main.get_put_latency(1) for _ in range(100)]

        assert all(lat >= main.MIN_LATENCY for lat in latencies)


class TestContentionScaling:
    """Test contention-based latency scaling."""

    def test_contention_disabled_by_default_without_provider(self, tmp_path):
        """Without provider, contention scaling should be disabled."""
        config_path = write_temp_config(create_minimal_config(
            provider=None,
            extra_storage="""
T_CAS.mean = 50
T_CAS.stddev = 5
T_MANIFEST_LIST.read.mean = 30
T_MANIFEST_LIST.read.stddev = 3
T_MANIFEST_LIST.write.mean = 40
T_MANIFEST_LIST.write.stddev = 4
T_MANIFEST_FILE.read.mean = 30
T_MANIFEST_FILE.read.stddev = 3
T_MANIFEST_FILE.write.mean = 40
T_MANIFEST_FILE.write.stddev = 4
"""
        ))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            # Without provider, contention scaling is disabled by default
            assert main.CONTENTION_SCALING_ENABLED == False
        finally:
            os.unlink(config_path)

    def test_contention_enabled_with_provider(self, tmp_path):
        """With provider, contention scaling should be auto-enabled."""
        config_path = write_temp_config(create_minimal_config(provider="s3x"))
        try:
            main.STATS = Stats()
            main.configure_from_toml(config_path)

            assert main.CONTENTION_SCALING_ENABLED == True
        finally:
            os.unlink(config_path)


class TestAppendSupport:
    """Test provider append support configuration."""

    def test_s3_no_append_support(self):
        """S3 Standard should not support append."""
        s3 = main.PROVIDER_PROFILES['s3']
        assert s3['append'] is None

    def test_s3x_has_append_support(self):
        """S3 Express should support append."""
        s3x = main.PROVIDER_PROFILES['s3x']
        assert s3x['append'] is not None
        assert 'median' in s3x['append']

    def test_azure_has_append_support(self):
        """Azure Blob should support append."""
        azure = main.PROVIDER_PROFILES['azure']
        assert azure['append'] is not None

    def test_gcp_no_append_support(self):
        """GCP should not support append."""
        gcp = main.PROVIDER_PROFILES['gcp']
        assert gcp['append'] is None

    def test_instant_has_append_support(self):
        """Instant provider should support append."""
        instant = main.PROVIDER_PROFILES['instant']
        assert instant['append'] is not None
