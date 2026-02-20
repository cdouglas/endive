"""Catalog provider abstraction for Iceberg table pointer operations.

This module separates catalog latencies from storage latencies. The catalog
is where the table pointer (metadata location) lives and supports CAS operations.
Storage is where the actual data (manifest lists, manifest files, data files) lives.

Valid configurations:
- InstantCatalog + S3 storage: Fast catalog (Nessie) with S3 data
- ObjectStorageCatalog(s3x) + S3 storage: S3 Express for both (conditional writes)
- ObjectStorageCatalog(azure) + Azure storage: Azure Premium for both

References:
- Iceberg catalogs: https://iceberg.apache.org/concepts/catalog/
- S3 conditional writes: PutObject with If-None-Match
- Azure conditional writes: ETag-based optimistic concurrency
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator
import numpy as np


@dataclass(frozen=True)
class CatalogConfig:
    """Configuration for a catalog provider.

    Attributes:
        cas_median: Median CAS latency in milliseconds
        cas_sigma: Log-normal sigma for CAS latency distribution
        min_latency: Minimum possible latency (network floor)
        supports_append: Whether catalog supports append operations
        append_median: Median append latency (if supported)
        append_sigma: Log-normal sigma for append latency
    """
    cas_median: float
    cas_sigma: float
    min_latency: float
    supports_append: bool = False
    append_median: float | None = None
    append_sigma: float | None = None


# Pre-defined catalog configurations
CATALOG_CONFIGS = {
    # Instant catalog: models fast catalog services (Nessie, Polaris on fast infra)
    "instant": CatalogConfig(
        cas_median=1.0,
        cas_sigma=0.1,
        min_latency=1.0,
        supports_append=True,
        append_median=1.0,
        append_sigma=0.1,
    ),

    # S3 Express One Zone: conditional PutObject
    # Source: YCSB measurements, CAS min=12.9ms, median=22ms
    "s3x": CatalogConfig(
        cas_median=22.0,
        cas_sigma=0.5,
        min_latency=10.0,
        supports_append=True,
        append_median=21.0,
        append_sigma=0.5,
    ),

    # S3 Standard: conditional PutObject (higher latency)
    # Source: YCSB measurements, CAS min=42.7ms, median=61ms
    "s3": CatalogConfig(
        cas_median=61.0,
        cas_sigma=0.5,
        min_latency=43.0,
        supports_append=False,  # S3 Standard doesn't support append
    ),

    # Azure Premium Blob: ETag-based conditional writes
    # Source: YCSB measurements, CAS min=46.1ms, median=64ms
    "azurex": CatalogConfig(
        cas_median=64.0,
        cas_sigma=0.5,
        min_latency=40.0,
        supports_append=True,
        append_median=70.0,
        append_sigma=0.5,
    ),

    # Azure Standard Blob
    # Source: YCSB measurements, CAS min=58.2ms, median=93ms
    "azure": CatalogConfig(
        cas_median=93.0,
        cas_sigma=0.5,
        min_latency=51.0,
        supports_append=True,
        append_median=87.0,
        append_sigma=0.5,
    ),

    # GCS: conditional writes via generation numbers
    # Source: YCSB measurements, CAS min=117.7ms, median=170ms
    "gcp": CatalogConfig(
        cas_median=170.0,
        cas_sigma=0.5,
        min_latency=118.0,
        supports_append=False,
    ),
}


class CatalogProvider(ABC):
    """Abstract interface for catalog operations.

    A catalog provider handles the table pointer (metadata location) and
    supports compare-and-swap (CAS) operations for atomic updates.

    This is separate from storage providers which handle manifest list
    and manifest file I/O.
    """

    @abstractmethod
    def get_cas_latency(self) -> float:
        """Generate a CAS operation latency in milliseconds."""
        pass

    @abstractmethod
    def get_append_latency(self) -> float:
        """Generate an append operation latency in milliseconds.

        Raises:
            NotImplementedError: If catalog doesn't support append operations
        """
        pass

    @abstractmethod
    def supports_append(self) -> bool:
        """Whether this catalog supports append operations."""
        pass

    @abstractmethod
    def get_min_latency(self) -> float:
        """Minimum possible latency (network floor)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this catalog provider."""
        pass


class InstantCatalog(CatalogProvider):
    """Instant catalog with ~1ms latency.

    Models fast catalog services like Nessie or Polaris running on
    low-latency infrastructure. Useful for isolating storage effects
    in experiments.
    """

    def __init__(self, config: CatalogConfig | None = None):
        self._config = config or CATALOG_CONFIGS["instant"]

    def get_cas_latency(self) -> float:
        latency = np.random.lognormal(
            mean=np.log(self._config.cas_median),
            sigma=self._config.cas_sigma
        )
        return max(self._config.min_latency, latency)

    def get_append_latency(self) -> float:
        if not self._config.supports_append:
            raise NotImplementedError("Instant catalog doesn't support append")
        latency = np.random.lognormal(
            mean=np.log(self._config.append_median),
            sigma=self._config.append_sigma
        )
        return max(self._config.min_latency, latency)

    def supports_append(self) -> bool:
        return self._config.supports_append

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return "instant"


class ObjectStorageCatalog(CatalogProvider):
    """Catalog backed by object storage conditional writes.

    Models catalogs that use object storage for the table pointer,
    using conditional writes (S3 If-None-Match, Azure ETag) for CAS.

    Examples:
    - S3 tables with conditional PutObject for metadata
    - Azure tables with ETag-based optimistic concurrency
    - GCS tables with generation-based conditions
    """

    def __init__(self, provider: str):
        """Initialize with a storage provider name.

        Args:
            provider: One of 's3x', 's3', 'azurex', 'azure', 'gcp'

        Raises:
            ValueError: If provider is not recognized
        """
        if provider not in CATALOG_CONFIGS:
            raise ValueError(f"Unknown catalog provider: {provider}. "
                           f"Valid options: {list(CATALOG_CONFIGS.keys())}")
        self._provider = provider
        self._config = CATALOG_CONFIGS[provider]

    def get_cas_latency(self) -> float:
        latency = np.random.lognormal(
            mean=np.log(self._config.cas_median),
            sigma=self._config.cas_sigma
        )
        return max(self._config.min_latency, latency)

    def get_append_latency(self) -> float:
        if not self._config.supports_append:
            raise NotImplementedError(
                f"{self._provider} catalog doesn't support append operations"
            )
        latency = np.random.lognormal(
            mean=np.log(self._config.append_median),
            sigma=self._config.append_sigma
        )
        return max(self._config.min_latency, latency)

    def supports_append(self) -> bool:
        return self._config.supports_append

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return self._provider


def create_catalog_provider(catalog_type: str, storage_provider: str | None = None) -> CatalogProvider:
    """Factory function to create a catalog provider.

    Args:
        catalog_type: Either 'instant' or 'object_storage'
        storage_provider: For object_storage type, the provider name
                         (s3x, s3, azurex, azure, gcp)

    Returns:
        CatalogProvider instance

    Examples:
        # Fast catalog for baseline experiments
        catalog = create_catalog_provider('instant')

        # S3 Express as catalog (conditional writes)
        catalog = create_catalog_provider('object_storage', 's3x')

        # Azure Premium as catalog
        catalog = create_catalog_provider('object_storage', 'azurex')
    """
    if catalog_type == "instant":
        return InstantCatalog()
    elif catalog_type == "object_storage":
        if storage_provider is None:
            raise ValueError("storage_provider required for object_storage catalog")
        return ObjectStorageCatalog(storage_provider)
    else:
        raise ValueError(f"Unknown catalog type: {catalog_type}. "
                        f"Valid options: 'instant', 'object_storage'")
