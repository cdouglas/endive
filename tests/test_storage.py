"""Tests for endive.storage — the new generator-based StorageProvider per SPEC.md §1.

Unit tests:
- StorageResult is immutable (frozen dataclass)
- Each operation returns StorageResult with correct fields
- Latency >= min_latency for each provider
- UnsupportedOperationError for unavailable operations
- Deterministic with seeded RNG
- LatencyDistribution implementations produce correct ranges
- Factory creates correct provider types
"""

import pytest
import numpy as np

from endive.storage import (
    StorageResult,
    UnsupportedOperationError,
    LatencyDistribution,
    LognormalLatency,
    SizeBasedLatency,
    FixedLatency,
    StorageProvider,
    S3StorageProvider,
    S3ExpressStorageProvider,
    AzureBlobStorageProvider,
    GCPStorageProvider,
    InstantStorageProvider,
    create_provider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def exhaust_generator(gen):
    """Drive a storage operation generator to completion, return result.

    Simulates SimPy: each yielded value is a timeout we'd wait for.
    We just advance and collect the return value.
    """
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def collect_latencies(gen_factory, n=100):
    """Collect n latency yields from repeated generator calls."""
    latencies = []
    for _ in range(n):
        gen = gen_factory()
        try:
            latency = next(gen)
            latencies.append(latency)
            # Exhaust remaining (should StopIteration with result)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass
        except StopIteration:
            pass
    return latencies


# ---------------------------------------------------------------------------
# StorageResult
# ---------------------------------------------------------------------------

class TestStorageResult:
    """StorageResult is a frozen (immutable) dataclass."""

    def test_fields_accessible(self):
        r = StorageResult(success=True, latency_ms=10.5, data_size_bytes=1024)
        assert r.success is True
        assert r.latency_ms == 10.5
        assert r.data_size_bytes == 1024

    def test_immutable(self):
        r = StorageResult(success=True, latency_ms=10.0, data_size_bytes=100)
        with pytest.raises(AttributeError):
            r.success = False
        with pytest.raises(AttributeError):
            r.latency_ms = 999.0

    def test_equality(self):
        a = StorageResult(success=True, latency_ms=5.0, data_size_bytes=50)
        b = StorageResult(success=True, latency_ms=5.0, data_size_bytes=50)
        assert a == b

    def test_inequality(self):
        a = StorageResult(success=True, latency_ms=5.0, data_size_bytes=50)
        b = StorageResult(success=False, latency_ms=5.0, data_size_bytes=50)
        assert a != b


# ---------------------------------------------------------------------------
# LatencyDistribution implementations
# ---------------------------------------------------------------------------

class TestLognormalLatency:
    """LognormalLatency samples from lognormal with floor."""

    def test_samples_positive(self):
        rng = np.random.RandomState(42)
        dist = LognormalLatency.from_median(median_ms=50.0, sigma=0.3, min_latency_ms=10.0)
        for _ in range(200):
            assert dist.sample(rng) >= 10.0

    def test_median_approximately_correct(self):
        rng = np.random.RandomState(42)
        dist = LognormalLatency.from_median(median_ms=50.0, sigma=0.3, min_latency_ms=1.0)
        samples = [dist.sample(rng) for _ in range(5000)]
        median = np.median(samples)
        assert 40.0 < median < 60.0, f"Median {median} not near 50ms"

    def test_min_latency_enforced(self):
        rng = np.random.RandomState(42)
        # Very low median with high min should be clamped
        dist = LognormalLatency.from_median(median_ms=0.1, sigma=0.1, min_latency_ms=10.0)
        for _ in range(100):
            assert dist.sample(rng) >= 10.0

    def test_from_median_constructor(self):
        dist = LognormalLatency.from_median(median_ms=22.0, sigma=0.22, min_latency_ms=10.0)
        assert abs(dist.mu - np.log(22.0)) < 1e-10
        assert dist.sigma == 0.22
        assert dist.min_latency_ms == 10.0

    def test_frozen(self):
        dist = LognormalLatency(mu=3.0, sigma=0.3, min_latency_ms=1.0)
        with pytest.raises(AttributeError):
            dist.mu = 999.0


class TestSizeBasedLatency:
    """SizeBasedLatency scales with data size."""

    def test_larger_size_higher_latency(self):
        rng = np.random.RandomState(42)
        dist = SizeBasedLatency(base_latency_ms=30.0, latency_per_mib_ms=20.0,
                                sigma=0.1, min_latency_ms=1.0)

        small = [dist.with_size(1 * 1024 * 1024).sample(rng) for _ in range(500)]
        large = [dist.with_size(10 * 1024 * 1024).sample(rng) for _ in range(500)]

        assert np.median(large) > np.median(small)

    def test_min_latency_enforced(self):
        rng = np.random.RandomState(42)
        dist = SizeBasedLatency(base_latency_ms=0.01, latency_per_mib_ms=0.001,
                                sigma=0.1, min_latency_ms=10.0)
        for _ in range(100):
            assert dist.with_size(1).sample(rng) >= 10.0

    def test_with_size_returns_new_instance(self):
        dist = SizeBasedLatency(base_latency_ms=30.0, latency_per_mib_ms=20.0,
                                sigma=0.3, min_latency_ms=1.0)
        sized = dist.with_size(1024)
        assert sized is not dist
        assert sized._size_bytes == 1024
        assert dist._size_bytes == 0  # Original unchanged

    def test_zero_sigma_deterministic(self):
        rng = np.random.RandomState(42)
        dist = SizeBasedLatency(base_latency_ms=30.0, latency_per_mib_ms=20.0,
                                sigma=0.0, min_latency_ms=1.0)
        sized = dist.with_size(5 * 1024 * 1024)  # 5 MiB
        expected = 30.0 + 5.0 * 20.0  # 130ms
        for _ in range(10):
            assert sized.sample(rng) == expected


class TestFixedLatency:
    """FixedLatency always returns the same value."""

    def test_deterministic(self):
        rng = np.random.RandomState(42)
        dist = FixedLatency(latency_ms=5.0)
        for _ in range(100):
            assert dist.sample(rng) == 5.0


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

class TestProviderCapabilities:
    """Capability flags match SPEC.md §1.3 capabilities table."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    def test_s3_capabilities(self, rng):
        p = create_provider("s3", rng)
        assert p.supports_cas is True
        assert p.supports_append is False
        assert p.supports_tail_append is False

    def test_s3x_capabilities(self, rng):
        p = create_provider("s3x", rng)
        assert p.supports_cas is True
        assert p.supports_append is True
        assert p.supports_tail_append is False

    def test_azure_capabilities(self, rng):
        p = create_provider("azure", rng)
        assert p.supports_cas is True
        assert p.supports_append is True
        assert p.supports_tail_append is False

    def test_azurex_capabilities(self, rng):
        p = create_provider("azurex", rng)
        assert p.supports_cas is True
        assert p.supports_append is True
        assert p.supports_tail_append is False

    def test_gcp_capabilities(self, rng):
        p = create_provider("gcp", rng)
        assert p.supports_cas is True
        assert p.supports_append is False
        assert p.supports_tail_append is False

    def test_instant_capabilities(self, rng):
        p = create_provider("instant", rng)
        assert p.supports_cas is True
        assert p.supports_append is True
        assert p.supports_tail_append is True


# ---------------------------------------------------------------------------
# UnsupportedOperationError
# ---------------------------------------------------------------------------

class TestUnsupportedOperations:
    """Unsupported operations raise UnsupportedOperationError."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    def test_s3_append_raises(self, rng):
        p = create_provider("s3", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.append("key", 0, 100))

    def test_s3_tail_append_raises(self, rng):
        p = create_provider("s3", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.tail_append("key", 100))

    def test_s3x_tail_append_raises(self, rng):
        p = create_provider("s3x", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.tail_append("key", 100))

    def test_gcp_append_raises(self, rng):
        p = create_provider("gcp", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.append("key", 0, 100))

    def test_gcp_tail_append_raises(self, rng):
        p = create_provider("gcp", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.tail_append("key", 100))

    def test_azure_tail_append_raises(self, rng):
        p = create_provider("azure", rng)
        with pytest.raises(UnsupportedOperationError):
            exhaust_generator(p.tail_append("key", 100))


# ---------------------------------------------------------------------------
# Generator protocol (yields latency, returns StorageResult)
# ---------------------------------------------------------------------------

class TestGeneratorProtocol:
    """Operations are generators that yield latency and return StorageResult."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    def _run_op(self, gen):
        """Run a generator, return (yielded_latency, result)."""
        latency = next(gen)
        try:
            next(gen)
            pytest.fail("Generator should have returned after one yield")
        except StopIteration as e:
            return latency, e.value

    def test_read_yields_latency_returns_result(self, rng):
        p = create_provider("s3x", rng)
        latency, result = self._run_op(p.read("key", 1024))
        assert isinstance(latency, float)
        assert latency > 0
        assert isinstance(result, StorageResult)
        assert result.success is True
        assert result.data_size_bytes == 1024
        assert result.latency_ms == latency

    def test_write_yields_latency_returns_result(self, rng):
        p = create_provider("s3x", rng)
        latency, result = self._run_op(p.write("key", 2048))
        assert isinstance(latency, float)
        assert latency > 0
        assert isinstance(result, StorageResult)
        assert result.success is True
        assert result.data_size_bytes == 2048

    def test_cas_yields_latency_returns_result(self, rng):
        p = create_provider("s3x", rng)
        latency, result = self._run_op(p.cas("key", 0, 512))
        assert isinstance(latency, float)
        assert latency > 0
        assert isinstance(result, StorageResult)
        assert result.success is True

    def test_append_yields_latency_returns_result(self, rng):
        p = create_provider("s3x", rng)
        latency, result = self._run_op(p.append("key", 0, 256))
        assert isinstance(latency, float)
        assert latency > 0
        assert isinstance(result, StorageResult)
        assert result.success is True

    def test_tail_append_yields_latency_returns_result(self, rng):
        p = create_provider("instant", rng)
        latency, result = self._run_op(p.tail_append("key", 128))
        assert isinstance(latency, float)
        assert latency > 0
        assert isinstance(result, StorageResult)
        assert result.success is True


# ---------------------------------------------------------------------------
# Minimum latency enforcement
# ---------------------------------------------------------------------------

class TestMinLatency:
    """All operations respect the provider's minimum latency floor."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.mark.parametrize("provider_name", ["instant", "s3x", "s3", "azure", "azurex", "gcp"])
    def test_read_respects_min_latency(self, provider_name, rng):
        p = create_provider(provider_name, rng)
        for _ in range(50):
            latency, _ = self._run_op(p.read("k", 100))
            assert latency >= p.min_latency_ms

    @pytest.mark.parametrize("provider_name", ["instant", "s3x", "s3", "azure", "azurex", "gcp"])
    def test_cas_respects_min_latency(self, provider_name, rng):
        p = create_provider(provider_name, rng)
        for _ in range(50):
            latency, _ = self._run_op(p.cas("k", 0, 100))
            assert latency >= p.min_latency_ms

    @pytest.mark.parametrize("provider_name", ["s3x", "azure", "azurex", "instant"])
    def test_append_respects_min_latency(self, provider_name, rng):
        p = create_provider(provider_name, rng)
        for _ in range(50):
            latency, _ = self._run_op(p.append("k", 0, 100))
            assert latency >= p.min_latency_ms

    def _run_op(self, gen):
        latency = next(gen)
        try:
            next(gen)
        except StopIteration:
            pass
        return latency, None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same seed produces identical latency sequences."""

    def test_same_seed_same_results(self):
        """Two providers with same seed produce identical latency sequences."""
        results1 = self._collect_latencies(seed=42)
        results2 = self._collect_latencies(seed=42)
        assert results1 == results2

    def test_different_seed_different_results(self):
        """Different seeds produce different sequences."""
        results1 = self._collect_latencies(seed=42)
        results2 = self._collect_latencies(seed=99)
        assert results1 != results2

    def _collect_latencies(self, seed: int) -> list:
        rng = np.random.RandomState(seed)
        p = create_provider("s3x", rng)
        latencies = []
        for _ in range(20):
            gen = p.read("key", 100)
            latency = next(gen)
            latencies.append(latency)
            try:
                next(gen)
            except StopIteration:
                pass
        return latencies


# ---------------------------------------------------------------------------
# Provider name property
# ---------------------------------------------------------------------------

class TestProviderNames:
    """Each provider reports its correct name."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.mark.parametrize("provider_name", ["instant", "s3", "s3x", "azure", "azurex", "gcp"])
    def test_name_matches(self, provider_name, rng):
        p = create_provider(provider_name, rng)
        assert p.name == provider_name


# ---------------------------------------------------------------------------
# Latency ranges (from YCSB benchmarks)
# ---------------------------------------------------------------------------

class TestLatencyRanges:
    """Provider latencies fall in expected ranges from YCSB measurements."""

    @pytest.mark.parametrize("provider_name,expected_min,expected_max", [
        ("instant", 0.5, 3.0),
        ("s3x", 10, 50),
        ("s3", 40, 120),
        ("azure", 50, 200),
        ("gcp", 100, 350),
    ])
    def test_read_latency_range(self, provider_name, expected_min, expected_max):
        rng = np.random.RandomState(42)
        p = create_provider(provider_name, rng)
        samples = []
        for _ in range(500):
            gen = p.read("key", 1024)
            latency = next(gen)
            samples.append(latency)
            try:
                next(gen)
            except StopIteration:
                pass
        median = np.median(samples)
        assert expected_min < median < expected_max, \
            f"{provider_name} read median {median:.1f} not in [{expected_min}, {expected_max}]"

    @pytest.mark.parametrize("provider_name,expected_min,expected_max", [
        ("instant", 0.5, 3.0),
        ("s3x", 10, 50),
        ("s3", 40, 120),
        ("azure", 50, 200),
        ("gcp", 100, 350),
    ])
    def test_cas_latency_range(self, provider_name, expected_min, expected_max):
        rng = np.random.RandomState(42)
        p = create_provider(provider_name, rng)
        samples = []
        for _ in range(500):
            gen = p.cas("key", 0, 100)
            latency = next(gen)
            samples.append(latency)
            try:
                next(gen)
            except StopIteration:
                pass
        median = np.median(samples)
        assert expected_min < median < expected_max, \
            f"{provider_name} CAS median {median:.1f} not in [{expected_min}, {expected_max}]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    """create_provider() builds correct provider types."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    def test_creates_s3(self, rng):
        p = create_provider("s3", rng)
        assert isinstance(p, S3StorageProvider)

    def test_creates_s3x(self, rng):
        p = create_provider("s3x", rng)
        assert isinstance(p, S3ExpressStorageProvider)

    def test_creates_azure(self, rng):
        p = create_provider("azure", rng)
        assert isinstance(p, AzureBlobStorageProvider)

    def test_creates_azurex(self, rng):
        p = create_provider("azurex", rng)
        assert isinstance(p, AzureBlobStorageProvider)

    def test_creates_gcp(self, rng):
        p = create_provider("gcp", rng)
        assert isinstance(p, GCPStorageProvider)

    def test_creates_instant(self, rng):
        p = create_provider("instant", rng)
        assert isinstance(p, InstantStorageProvider)

    def test_invalid_provider_raises(self, rng):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent", rng)

    def test_default_rng_created_if_none(self):
        """Factory works without explicit RNG."""
        p = create_provider("instant")
        assert isinstance(p, InstantStorageProvider)

    def test_aws_alias(self, rng):
        """'aws' is an alias for 's3x' in PROVIDER_PROFILES."""
        p = create_provider("aws", rng)
        # aws maps to s3x profile, which creates S3ExpressStorageProvider
        assert isinstance(p, S3ExpressStorageProvider)


# ---------------------------------------------------------------------------
# Min latency values (from MEMORY.md / config.py)
# ---------------------------------------------------------------------------

class TestMinLatencyValues:
    """Provider min_latency_ms matches documented values."""

    @pytest.fixture
    def rng(self):
        return np.random.RandomState(42)

    @pytest.mark.parametrize("provider_name,expected_min_latency", [
        ("instant", 1),
        ("s3x", 10),
        ("s3", 43),
        ("azurex", 40),
        ("azure", 51),
        ("gcp", 118),
    ])
    def test_min_latency_value(self, provider_name, expected_min_latency, rng):
        p = create_provider(provider_name, rng)
        assert p.min_latency_ms == expected_min_latency


# ---------------------------------------------------------------------------
# Provider ordering (relative latencies)
# ---------------------------------------------------------------------------

class TestProviderOrdering:
    """Providers ordered by latency: instant < s3x < s3 < azure < gcp."""

    def test_s3_slower_than_s3x(self):
        rng = np.random.RandomState(42)
        s3 = create_provider("s3", rng)
        s3x = create_provider("s3x", np.random.RandomState(42))

        s3_samples = [next(s3.cas("k", 0, 100)) for _ in range(500)]
        s3x_samples = [next(s3x.cas("k", 0, 100)) for _ in range(500)]
        # Exhaust generators
        for gen_val in [s3.cas("k", 0, 100) for _ in range(0)]:
            pass

        assert np.median(s3_samples) > np.median(s3x_samples)

    def test_gcp_slowest(self):
        providers = {}
        for name in ["s3", "s3x", "azure", "gcp"]:
            rng = np.random.RandomState(42)
            p = create_provider(name, rng)
            samples = []
            for _ in range(500):
                gen = p.cas("k", 0, 100)
                samples.append(next(gen))
                try:
                    next(gen)
                except StopIteration:
                    pass
            providers[name] = np.median(samples)

        for other in ["s3", "s3x", "azure"]:
            assert providers["gcp"] > providers[other], \
                f"GCP ({providers['gcp']:.1f}) not slower than {other} ({providers[other]:.1f})"


# ---------------------------------------------------------------------------
# InstantStorageProvider specifics
# ---------------------------------------------------------------------------

class TestInstantProvider:
    """InstantStorageProvider has fixed configurable latency."""

    def test_default_1ms(self):
        rng = np.random.RandomState(42)
        p = InstantStorageProvider(rng=rng)
        gen = p.read("key", 100)
        latency = next(gen)
        assert latency == 1.0

    def test_custom_latency(self):
        rng = np.random.RandomState(42)
        p = InstantStorageProvider(rng=rng, latency_ms=5.0)
        gen = p.cas("key", 0, 100)
        latency = next(gen)
        assert latency == 5.0

    def test_all_operations_same_latency(self):
        rng = np.random.RandomState(42)
        p = InstantStorageProvider(rng=rng, latency_ms=2.0)

        ops = [
            p.read("k", 100),
            p.write("k", 100),
            p.cas("k", 0, 100),
            p.append("k", 0, 100),
            p.tail_append("k", 100),
        ]
        for gen in ops:
            assert next(gen) == 2.0
