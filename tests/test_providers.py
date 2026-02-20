"""Tests for catalog and storage provider implementations."""

import pytest
import numpy as np
from endive.catalog_provider import (
    CatalogProvider,
    CatalogConfig,
    InstantCatalog,
    ObjectStorageCatalog,
    create_catalog_provider,
    CATALOG_CONFIGS,
)
from endive.storage_provider import (
    StorageProvider,
    StorageConfig,
    ObjectStorageProvider,
    create_storage_provider,
    STORAGE_CONFIGS,
)


class TestCatalogConfigs:
    """Test catalog configuration definitions."""

    def test_all_providers_have_configs(self):
        """All expected providers have configurations."""
        expected = {"instant", "s3x", "s3", "azurex", "azure", "gcp", "aws"}
        assert expected == set(CATALOG_CONFIGS.keys())

    def test_instant_config_values(self):
        """Instant catalog has ~1ms latencies."""
        cfg = CATALOG_CONFIGS["instant"]
        assert cfg.cas_median == 1.0
        assert cfg.min_latency == 1.0
        assert cfg.supports_append is True

    def test_s3x_config_values(self):
        """S3X catalog has realistic latencies from YCSB."""
        cfg = CATALOG_CONFIGS["s3x"]
        assert cfg.cas_median == 22.0
        assert cfg.min_latency == 10.0
        assert cfg.supports_append is True

    def test_s3_does_not_support_append(self):
        """S3 Standard doesn't support append operations."""
        cfg = CATALOG_CONFIGS["s3"]
        assert cfg.supports_append is False

    def test_configs_have_positive_latencies(self):
        """All configs have positive latencies."""
        for name, cfg in CATALOG_CONFIGS.items():
            assert cfg.cas_median > 0, f"{name} has non-positive cas_median"
            assert cfg.min_latency > 0, f"{name} has non-positive min_latency"
            if cfg.supports_append:
                assert cfg.append_median > 0, f"{name} has non-positive append_median"


class TestInstantCatalog:
    """Test InstantCatalog implementation."""

    def test_get_cas_latency_returns_positive(self):
        """CAS latency is always positive."""
        catalog = InstantCatalog()
        for _ in range(100):
            latency = catalog.get_cas_latency()
            assert latency > 0

    def test_get_cas_latency_near_1ms(self):
        """CAS latency is approximately 1ms median."""
        np.random.seed(42)
        catalog = InstantCatalog()
        samples = [catalog.get_cas_latency() for _ in range(1000)]
        median = np.median(samples)
        assert 0.5 < median < 2.0, f"Median {median} not near 1ms"

    def test_get_cas_latency_respects_min(self):
        """CAS latency respects minimum latency floor."""
        catalog = InstantCatalog()
        min_latency = catalog.get_min_latency()
        for _ in range(100):
            latency = catalog.get_cas_latency()
            assert latency >= min_latency

    def test_supports_append(self):
        """Instant catalog supports append."""
        catalog = InstantCatalog()
        assert catalog.supports_append() is True

    def test_get_append_latency(self):
        """Append latency works when supported."""
        catalog = InstantCatalog()
        latency = catalog.get_append_latency()
        assert latency > 0

    def test_name_property(self):
        """Name property returns 'instant'."""
        catalog = InstantCatalog()
        assert catalog.name == "instant"


class TestObjectStorageCatalog:
    """Test ObjectStorageCatalog implementation."""

    def test_create_with_valid_provider(self):
        """Can create catalog with valid provider names."""
        for provider in ["s3x", "s3", "azurex", "azure", "gcp", "instant"]:
            catalog = ObjectStorageCatalog(provider)
            assert catalog.name == provider

    def test_create_with_invalid_provider_raises(self):
        """Invalid provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown catalog provider"):
            ObjectStorageCatalog("invalid_provider")

    def test_s3x_cas_latency_distribution(self):
        """S3X CAS latency has correct distribution."""
        np.random.seed(42)
        catalog = ObjectStorageCatalog("s3x")
        samples = [catalog.get_cas_latency() for _ in range(1000)]
        median = np.median(samples)
        # S3X median should be ~22ms
        assert 15 < median < 35, f"S3X median {median} not near 22ms"

    def test_s3_cas_latency_higher_than_s3x(self):
        """S3 Standard has higher latency than S3 Express."""
        np.random.seed(42)
        s3x = ObjectStorageCatalog("s3x")
        s3 = ObjectStorageCatalog("s3")

        s3x_samples = [s3x.get_cas_latency() for _ in range(500)]
        s3_samples = [s3.get_cas_latency() for _ in range(500)]

        assert np.median(s3_samples) > np.median(s3x_samples)

    def test_s3_append_not_supported(self):
        """S3 Standard doesn't support append."""
        catalog = ObjectStorageCatalog("s3")
        assert catalog.supports_append() is False
        with pytest.raises(NotImplementedError):
            catalog.get_append_latency()

    def test_azure_append_supported(self):
        """Azure supports append operations."""
        catalog = ObjectStorageCatalog("azure")
        assert catalog.supports_append() is True
        latency = catalog.get_append_latency()
        assert latency > 0

    def test_min_latency_enforced(self):
        """Minimum latency is enforced."""
        catalog = ObjectStorageCatalog("s3x")
        min_lat = catalog.get_min_latency()
        for _ in range(100):
            assert catalog.get_cas_latency() >= min_lat


class TestCreateCatalogProvider:
    """Test catalog provider factory function."""

    def test_create_instant(self):
        """Create instant catalog."""
        catalog = create_catalog_provider("instant")
        assert isinstance(catalog, InstantCatalog)

    def test_create_object_storage(self):
        """Create object storage catalog."""
        catalog = create_catalog_provider("object_storage", "s3x")
        assert isinstance(catalog, ObjectStorageCatalog)
        assert catalog.name == "s3x"

    def test_object_storage_requires_provider(self):
        """Object storage type requires provider name."""
        with pytest.raises(ValueError, match="storage_provider required"):
            create_catalog_provider("object_storage")

    def test_invalid_type_raises(self):
        """Invalid catalog type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown catalog type"):
            create_catalog_provider("invalid_type")


class TestStorageConfigs:
    """Test storage configuration definitions."""

    def test_all_providers_have_configs(self):
        """All expected providers have configurations."""
        expected = {"instant", "s3x", "s3", "azurex", "azure", "gcp", "aws"}
        assert expected == set(STORAGE_CONFIGS.keys())

    def test_instant_config_values(self):
        """Instant storage has ~1ms latencies."""
        cfg = STORAGE_CONFIGS["instant"]
        assert cfg.ml_read_median == 1.0
        assert cfg.ml_write_median == 1.0
        assert cfg.min_latency == 1.0

    def test_s3x_config_values(self):
        """S3X storage has realistic latencies."""
        cfg = STORAGE_CONFIGS["s3x"]
        assert cfg.ml_read_median == 22.0
        assert cfg.ml_write_median == 21.0
        assert cfg.min_latency == 10.0


class TestObjectStorageProvider:
    """Test ObjectStorageProvider implementation."""

    def test_create_with_valid_provider(self):
        """Can create storage with valid provider names."""
        for provider in ["instant", "s3x", "s3", "azurex", "azure", "gcp"]:
            storage = ObjectStorageProvider(provider)
            assert storage.name == provider

    def test_create_with_invalid_provider_raises(self):
        """Invalid provider name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown storage provider"):
            ObjectStorageProvider("invalid_provider")

    def test_manifest_list_read_latency(self):
        """ML read latency is positive and reasonable."""
        np.random.seed(42)
        storage = ObjectStorageProvider("s3x")
        samples = [storage.get_manifest_list_read_latency() for _ in range(100)]
        assert all(s > 0 for s in samples)
        assert 10 < np.median(samples) < 50

    def test_manifest_list_write_latency(self):
        """ML write latency is positive and reasonable."""
        np.random.seed(42)
        storage = ObjectStorageProvider("s3x")
        samples = [storage.get_manifest_list_write_latency() for _ in range(100)]
        assert all(s > 0 for s in samples)
        assert 10 < np.median(samples) < 50

    def test_manifest_file_read_latency(self):
        """MF read latency is positive."""
        storage = ObjectStorageProvider("s3x")
        latency = storage.get_manifest_file_read_latency()
        assert latency > 0

    def test_manifest_file_write_latency(self):
        """MF write latency is positive."""
        storage = ObjectStorageProvider("s3x")
        latency = storage.get_manifest_file_write_latency()
        assert latency > 0

    def test_put_latency_scales_with_size(self):
        """PUT latency increases with file size."""
        np.random.seed(42)
        storage = ObjectStorageProvider("s3x")

        # Sample at different sizes
        small_samples = [storage.get_put_latency(0.1) for _ in range(100)]
        large_samples = [storage.get_put_latency(10.0) for _ in range(100)]

        # Large files should have higher median latency
        assert np.median(large_samples) > np.median(small_samples)

    def test_size_parameter_accepted(self):
        """Size parameters are accepted by all methods."""
        storage = ObjectStorageProvider("s3x")

        # These should not raise
        storage.get_manifest_list_read_latency(size_kib=50.0)
        storage.get_manifest_list_write_latency(size_kib=50.0)
        storage.get_manifest_file_read_latency(size_kib=200.0)
        storage.get_manifest_file_write_latency(size_kib=200.0)
        storage.get_put_latency(size_mib=5.0)

    def test_min_latency_enforced(self):
        """Minimum latency is enforced for all operations."""
        storage = ObjectStorageProvider("s3x")
        min_lat = storage.get_min_latency()

        for _ in range(50):
            assert storage.get_manifest_list_read_latency() >= min_lat
            assert storage.get_manifest_list_write_latency() >= min_lat
            assert storage.get_manifest_file_read_latency() >= min_lat
            assert storage.get_manifest_file_write_latency() >= min_lat
            assert storage.get_put_latency(1.0) >= min_lat


class TestCreateStorageProvider:
    """Test storage provider factory function."""

    def test_create_instant(self):
        """Create instant storage provider."""
        storage = create_storage_provider("instant")
        assert isinstance(storage, ObjectStorageProvider)
        assert storage.name == "instant"

    def test_create_s3x(self):
        """Create S3X storage provider."""
        storage = create_storage_provider("s3x")
        assert storage.name == "s3x"

    def test_invalid_provider_raises(self):
        """Invalid provider raises ValueError."""
        with pytest.raises(ValueError):
            create_storage_provider("invalid")


class TestProviderIntegration:
    """Integration tests for providers with main.py."""

    def test_providers_initialized_from_config(self):
        """Providers are initialized when loading config."""
        from endive.main import configure_from_toml
        from endive.main import CATALOG_PROVIDER_INSTANCE, STORAGE_PROVIDER_INSTANCE

        configure_from_toml("experiment_configs/exp1_fastappend_baseline.toml")

        # Re-import to get updated values
        from endive.main import CATALOG_PROVIDER_INSTANCE, STORAGE_PROVIDER_INSTANCE

        assert CATALOG_PROVIDER_INSTANCE is not None
        assert STORAGE_PROVIDER_INSTANCE is not None

    def test_latency_functions_use_providers(self):
        """Latency functions use provider instances."""
        from endive.main import configure_from_toml, get_cas_latency, get_manifest_list_latency
        import numpy as np

        np.random.seed(42)
        configure_from_toml("experiment_configs/validation/s3x_throughput_test.toml")

        # S3X should have ~22ms CAS
        cas_samples = [get_cas_latency() for _ in range(100)]
        median = np.median(cas_samples)
        assert 15 < median < 35, f"CAS median {median} not in S3X range"

        # S3X should have ~22ms ML read
        ml_samples = [get_manifest_list_latency("read") for _ in range(100)]
        ml_median = np.median(ml_samples)
        assert 15 < ml_median < 35, f"ML read median {ml_median} not in S3X range"

    def test_separated_catalog_storage_config(self):
        """Separated catalog and storage providers work."""
        from endive.main import configure_from_toml
        import numpy as np

        np.random.seed(42)
        configure_from_toml("experiment_configs/examples/separated_providers.toml")

        from endive.main import CATALOG_PROVIDER_INSTANCE, STORAGE_PROVIDER_INSTANCE
        from endive.main import get_cas_latency, get_manifest_list_latency

        # Instant catalog, S3X storage
        assert CATALOG_PROVIDER_INSTANCE.name == "instant"
        assert STORAGE_PROVIDER_INSTANCE.name == "s3x"

        # CAS should be ~1ms (instant)
        cas_samples = [get_cas_latency() for _ in range(100)]
        assert np.median(cas_samples) < 3.0

        # ML read should be ~22ms (s3x)
        ml_samples = [get_manifest_list_latency("read") for _ in range(100)]
        assert np.median(ml_samples) > 10.0


class TestProviderLatencyRanges:
    """Test that provider latencies match YCSB measurements."""

    @pytest.mark.parametrize("provider,expected_min,expected_max", [
        ("instant", 0.5, 3.0),
        ("s3x", 15, 40),
        ("s3", 45, 100),
        ("azure", 70, 150),
        ("gcp", 130, 250),
    ])
    def test_cas_latency_in_expected_range(self, provider, expected_min, expected_max):
        """CAS latency median falls in expected range."""
        np.random.seed(42)
        catalog = ObjectStorageCatalog(provider)
        samples = [catalog.get_cas_latency() for _ in range(500)]
        median = np.median(samples)
        assert expected_min < median < expected_max, \
            f"{provider} CAS median {median:.1f} not in [{expected_min}, {expected_max}]"

    @pytest.mark.parametrize("provider,expected_min,expected_max", [
        ("instant", 0.5, 3.0),
        ("s3x", 15, 40),
        ("s3", 45, 100),
        ("azure", 65, 130),
        ("gcp", 130, 250),
    ])
    def test_ml_read_latency_in_expected_range(self, provider, expected_min, expected_max):
        """ML read latency median falls in expected range."""
        np.random.seed(42)
        storage = ObjectStorageProvider(provider)
        samples = [storage.get_manifest_list_read_latency() for _ in range(500)]
        median = np.median(samples)
        assert expected_min < median < expected_max, \
            f"{provider} ML read median {median:.1f} not in [{expected_min}, {expected_max}]"
