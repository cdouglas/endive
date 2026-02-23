"""Storage provider abstraction per SPEC.md §1.

This module defines the new generator-based storage interface where all
operations yield SimPy timeouts. Latencies are drawn from opaque
LatencyDistribution objects provided at construction.

Key types:
- StorageResult: Immutable result of any storage operation
- LatencyDistribution: Protocol for opaque latency sampling
- LognormalLatency: Lognormal distribution with minimum floor
- SizeBasedLatency: Size-dependent latency model (base + rate * size)
- StorageProvider: ABC with read/write/cas/append/tail_append
- UnsupportedOperationError: Raised for unavailable operations

Concrete providers:
- S3StorageProvider: CAS only (no append)
- S3ExpressStorageProvider: CAS + append
- AzureBlobStorageProvider: CAS + append (Standard and Premium)
- GCPStorageProvider: CAS only (heavy tails)
- InstantStorageProvider: All operations, 1ms latency (testing)
"""

from __future__ import annotations

import tomllib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np


# ---------------------------------------------------------------------------
# Provider profile loading
# ---------------------------------------------------------------------------

_PROVIDERS_DIR = Path(__file__).parent / "providers"
_PROFILE_CACHE: dict[str, dict] = {}
_ALIASES: dict[str, str] = {"aws": "s3x"}

_VALID_PROVIDERS = frozenset({"s3", "s3x", "azure", "azurex", "gcp", "instant"})


def _load_provider_profile(name: str) -> dict:
    """Load a provider profile from TOML, with caching and alias resolution."""
    resolved = _ALIASES.get(name, name)
    if resolved in _PROFILE_CACHE:
        return _PROFILE_CACHE[resolved]

    toml_path = _PROVIDERS_DIR / f"{resolved}.toml"
    if not toml_path.exists():
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Valid: {sorted(_VALID_PROVIDERS | set(_ALIASES))}"
        )

    with open(toml_path, "rb") as f:
        profile = tomllib.load(f)

    _PROFILE_CACHE[resolved] = profile
    return profile


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageResult:
    """Immutable result of a storage operation."""
    success: bool
    latency_ms: float
    data_size_bytes: int


class UnsupportedOperationError(Exception):
    """Raised when a storage operation is not supported by the provider."""
    pass


# ---------------------------------------------------------------------------
# Latency distributions
# ---------------------------------------------------------------------------

class LatencyDistribution(ABC):
    """Opaque latency distribution that samples values.

    All latency sampling goes through this interface. Inline conditional
    latency computation is prohibited by SPEC.md design principle 4.
    """

    @abstractmethod
    def sample(self, rng: np.random.RandomState) -> float:
        """Draw a latency sample in milliseconds.

        Args:
            rng: Seeded random state for determinism.

        Returns:
            Latency in milliseconds (always >= min_latency_ms).
        """
        ...


@dataclass(frozen=True)
class LognormalLatency(LatencyDistribution):
    """Lognormal distribution with minimum floor.

    Parameters are derived from YCSB benchmark measurements:
    - mu = ln(median) where median is the observed median latency
    - sigma controls tail heaviness
    - min_latency_ms is the physical network floor
    """
    mu: float
    sigma: float
    min_latency_ms: float = 1.0

    def sample(self, rng: np.random.RandomState) -> float:
        raw = rng.lognormal(mean=self.mu, sigma=self.sigma)
        return max(raw, self.min_latency_ms)

    @classmethod
    def from_median(cls, median_ms: float, sigma: float,
                    min_latency_ms: float = 1.0) -> LognormalLatency:
        """Construct from median latency (convenience)."""
        return cls(mu=float(np.log(median_ms)), sigma=sigma,
                   min_latency_ms=min_latency_ms)


@dataclass(frozen=True)
class SizeBasedLatency(LatencyDistribution):
    """Size-dependent latency model: base + rate * size_mib + noise.

    Used for PUT operations where latency scales with object size.
    Based on Durner et al. VLDB 2023 measurements.
    """
    base_latency_ms: float
    latency_per_mib_ms: float
    sigma: float
    min_latency_ms: float = 1.0
    _size_bytes: int = 0

    def with_size(self, size_bytes: int) -> SizeBasedLatency:
        """Return a copy with the given data size set."""
        return SizeBasedLatency(
            base_latency_ms=self.base_latency_ms,
            latency_per_mib_ms=self.latency_per_mib_ms,
            sigma=self.sigma,
            min_latency_ms=self.min_latency_ms,
            _size_bytes=size_bytes,
        )

    def sample(self, rng: np.random.RandomState) -> float:
        size_mib = self._size_bytes / (1024 * 1024)
        deterministic = self.base_latency_ms + self.latency_per_mib_ms * size_mib
        if self.sigma > 0 and deterministic > 0:
            noisy = rng.lognormal(mean=np.log(deterministic), sigma=self.sigma)
        else:
            noisy = max(deterministic, 0.0)
        return max(noisy, self.min_latency_ms)


@dataclass(frozen=True)
class FixedLatency(LatencyDistribution):
    """Fixed (deterministic) latency. Useful for testing."""
    latency_ms: float

    def sample(self, rng: np.random.RandomState) -> float:
        return self.latency_ms


# ---------------------------------------------------------------------------
# StorageProvider ABC
# ---------------------------------------------------------------------------

class StorageProvider(ABC):
    """Abstract storage provider with latency-bearing operations.

    All operations are generators that yield SimPy timeouts (floats).
    Latencies are drawn from LatencyDistribution objects provided at
    construction. The provider holds a seeded RNG for determinism.
    """

    def __init__(self, rng: np.random.RandomState):
        self._rng = rng

    # -- Abstract operations --

    @abstractmethod
    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        """Read object from storage."""
        ...

    @abstractmethod
    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Write object to storage (unconditional PUT).

        Uses size-based latency model (Durner et al.).
        """
        ...

    @abstractmethod
    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Compare-and-swap operation.

        Raises:
            UnsupportedOperationError: If provider doesn't support CAS.
        """
        ...

    @abstractmethod
    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Conditional append at specific offset.

        Raises:
            UnsupportedOperationError: If provider doesn't support append.
        """
        ...

    @abstractmethod
    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        """Unconditional append to end of object.

        Raises:
            UnsupportedOperationError: If provider doesn't support tail_append.
        """
        ...

    # -- Capability flags --

    @property
    @abstractmethod
    def supports_cas(self) -> bool:
        """Whether this provider supports CAS operations."""
        ...

    @property
    @abstractmethod
    def supports_append(self) -> bool:
        """Whether this provider supports conditional append."""
        ...

    @property
    @abstractmethod
    def supports_tail_append(self) -> bool:
        """Whether this provider supports unconditional tail append."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...

    @property
    @abstractmethod
    def min_latency_ms(self) -> float:
        """Minimum possible latency (network floor)."""
        ...


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------

class S3StorageProvider(StorageProvider):
    """AWS S3 Standard storage provider.

    Supports: read, write, cas
    Does NOT support: append, tail_append
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        read_latency: LatencyDistribution,
        write_latency: SizeBasedLatency,
        cas_latency: LatencyDistribution,
        min_latency: float = 43.0,
    ):
        super().__init__(rng)
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._cas_latency = cas_latency
        self._min_latency = min_latency

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._read_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._write_latency.with_size(size_bytes).sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._cas_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("S3 Standard does not support conditional append")

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("S3 Standard does not support tail append")

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return False

    @property
    def supports_tail_append(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "s3"

    @property
    def min_latency_ms(self) -> float:
        return self._min_latency


class S3ExpressStorageProvider(StorageProvider):
    """AWS S3 Express One Zone storage provider.

    Supports: read, write, cas, append
    Does NOT support: tail_append
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        read_latency: LatencyDistribution,
        write_latency: SizeBasedLatency,
        cas_latency: LatencyDistribution,
        append_latency: LatencyDistribution,
        min_latency: float = 10.0,
    ):
        super().__init__(rng)
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._cas_latency = cas_latency
        self._append_latency = append_latency
        self._min_latency = min_latency

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._read_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._write_latency.with_size(size_bytes).sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._cas_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._append_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("S3 Express does not support tail append")

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return True

    @property
    def supports_tail_append(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "s3x"

    @property
    def min_latency_ms(self) -> float:
        return self._min_latency


class AzureBlobStorageProvider(StorageProvider):
    """Azure Blob Storage provider (Standard or Premium).

    Supports: read, write, cas, append
    Does NOT support: tail_append

    Note: Azure append failures have dramatically different latency
    distributions (2000ms+ median) compared to successes (~87ms).
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        read_latency: LatencyDistribution,
        write_latency: SizeBasedLatency,
        cas_latency: LatencyDistribution,
        append_latency: LatencyDistribution,
        min_latency: float = 51.0,
        provider_name: str = "azure",
    ):
        super().__init__(rng)
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._cas_latency = cas_latency
        self._append_latency = append_latency
        self._min_latency = min_latency
        self._name = provider_name

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._read_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._write_latency.with_size(size_bytes).sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._cas_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._append_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("Azure Blob does not support tail append")

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return True

    @property
    def supports_tail_append(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def min_latency_ms(self) -> float:
        return self._min_latency


class GCPStorageProvider(StorageProvider):
    """Google Cloud Storage provider.

    Supports: read, write, cas
    Does NOT support: append, tail_append

    Note: GCP has extremely heavy-tailed CAS failure latencies
    (median ~4500ms for failures vs ~170ms for successes).
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        read_latency: LatencyDistribution,
        write_latency: SizeBasedLatency,
        cas_latency: LatencyDistribution,
        min_latency: float = 118.0,
    ):
        super().__init__(rng)
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._cas_latency = cas_latency
        self._min_latency = min_latency

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._read_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._write_latency.with_size(size_bytes).sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        latency = self._cas_latency.sample(self._rng)
        yield latency
        return StorageResult(success=True, latency_ms=latency,
                             data_size_bytes=size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("GCP does not support conditional append")

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        raise UnsupportedOperationError("GCP does not support tail append")

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return False

    @property
    def supports_tail_append(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "gcp"

    @property
    def min_latency_ms(self) -> float:
        return self._min_latency


class InstantStorageProvider(StorageProvider):
    """Instant storage for testing (configurable fixed latency).

    Supports: read, write, cas, append, tail_append
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        latency_ms: float = 1.0,
    ):
        super().__init__(rng)
        self._latency = float(latency_ms)

    def _make_result(self, size_bytes: int) -> StorageResult:
        return StorageResult(success=True, latency_ms=self._latency,
                             data_size_bytes=size_bytes)

    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]:
        yield self._latency
        return self._make_result(expected_size_bytes)

    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        yield self._latency
        return self._make_result(size_bytes)

    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        yield self._latency
        return self._make_result(size_bytes)

    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]:
        yield self._latency
        return self._make_result(size_bytes)

    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]:
        yield self._latency
        return self._make_result(size_bytes)

    @property
    def supports_cas(self) -> bool:
        return True

    @property
    def supports_append(self) -> bool:
        return True

    @property
    def supports_tail_append(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "instant"

    @property
    def min_latency_ms(self) -> float:
        return self._latency


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_lognormal(profile: dict, section_name: str,
                     min_latency: float = 1.0) -> LognormalLatency:
    """Build a LognormalLatency from a provider TOML section."""
    section = profile[section_name]
    return LognormalLatency.from_median(
        median_ms=section["median_ms"],
        sigma=section["sigma"],
        min_latency_ms=min_latency,
    )


def _build_size_latency(profile: dict, min_latency: float = 1.0) -> SizeBasedLatency:
    """Build a SizeBasedLatency from a provider TOML [write] section."""
    w = profile["write"]
    return SizeBasedLatency(
        base_latency_ms=w["base_latency_ms"],
        latency_per_mib_ms=w["latency_per_mib_ms"],
        sigma=w["sigma"],
        min_latency_ms=min_latency,
    )


def create_provider(provider_name: str,
                    rng: np.random.RandomState | None = None) -> StorageProvider:
    """Factory function to create a StorageProvider from a provider name.

    Args:
        provider_name: One of 'instant', 's3', 's3x', 'azure', 'azurex', 'gcp', 'aws'.
        rng: Seeded random state. If None, creates unseeded one.

    Returns:
        Configured StorageProvider instance.
    """
    if rng is None:
        rng = np.random.RandomState()

    profile = _load_provider_profile(provider_name)
    resolved_name = _ALIASES.get(provider_name, provider_name)
    prov = profile["provider"]
    min_lat = prov["min_latency_ms"]

    if resolved_name == "instant":
        return InstantStorageProvider(rng=rng, latency_ms=min_lat)

    # Build common distributions
    read_latency = _build_lognormal(profile, "read", min_lat)
    write_latency = _build_size_latency(profile, min_lat)

    if resolved_name == "s3":
        return S3StorageProvider(
            rng=rng,
            read_latency=read_latency,
            write_latency=write_latency,
            cas_latency=_build_lognormal(profile, "cas", min_latency=min_lat),
            min_latency=min_lat,
        )

    if resolved_name == "s3x":
        return S3ExpressStorageProvider(
            rng=rng,
            read_latency=read_latency,
            write_latency=write_latency,
            cas_latency=_build_lognormal(profile, "cas", min_latency=min_lat),
            append_latency=_build_lognormal(profile, "append", min_latency=min_lat),
            min_latency=min_lat,
        )

    if resolved_name in ("azure", "azurex"):
        return AzureBlobStorageProvider(
            rng=rng,
            read_latency=read_latency,
            write_latency=write_latency,
            cas_latency=_build_lognormal(profile, "cas", min_latency=min_lat),
            append_latency=_build_lognormal(profile, "append", min_latency=min_lat),
            min_latency=min_lat,
            provider_name=resolved_name,
        )

    if resolved_name == "gcp":
        return GCPStorageProvider(
            rng=rng,
            read_latency=read_latency,
            write_latency=write_latency,
            cas_latency=_build_lognormal(profile, "cas", min_latency=min_lat),
            min_latency=min_lat,
        )

    raise ValueError(f"No provider implementation for {provider_name!r}")
