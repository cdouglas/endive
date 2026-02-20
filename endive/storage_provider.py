"""Storage provider abstraction for Iceberg data operations.

This module handles I/O for manifest lists, manifest files, and data files.
It is separate from the catalog provider which handles the table pointer.

Storage operations:
- Manifest list read/write
- Manifest file read/write
- Data file (Parquet) read/write
- PUT operations for new files

References:
- Iceberg file layout: https://iceberg.apache.org/spec/#file-system-operations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StorageConfig:
    """Configuration for a storage provider.

    All latencies in milliseconds.

    Attributes:
        ml_read_median: Manifest list read median latency
        ml_read_sigma: Log-normal sigma for ML read
        ml_write_median: Manifest list write median latency
        ml_write_sigma: Log-normal sigma for ML write
        mf_read_median: Manifest file read median latency
        mf_read_sigma: Log-normal sigma for MF read
        mf_write_median: Manifest file write median latency
        mf_write_sigma: Log-normal sigma for MF write
        put_base_latency: Base PUT latency (fixed overhead)
        put_per_mib: Additional latency per MiB of data
        put_sigma: Log-normal sigma for PUT
        min_latency: Minimum possible latency (network floor)
    """
    ml_read_median: float
    ml_read_sigma: float
    ml_write_median: float
    ml_write_sigma: float
    mf_read_median: float
    mf_read_sigma: float
    mf_write_median: float
    mf_write_sigma: float
    put_base_latency: float
    put_per_mib: float
    put_sigma: float
    min_latency: float


# Pre-defined storage configurations from YCSB measurements
STORAGE_CONFIGS = {
    # Instant storage for baseline experiments
    "instant": StorageConfig(
        ml_read_median=1.0,
        ml_read_sigma=0.1,
        ml_write_median=1.0,
        ml_write_sigma=0.1,
        mf_read_median=1.0,
        mf_read_sigma=0.1,
        mf_write_median=1.0,
        mf_write_sigma=0.1,
        put_base_latency=0.5,
        put_per_mib=0.1,
        put_sigma=0.1,
        min_latency=1.0,
    ),

    # S3 Express One Zone
    # Source: YCSB June 2025
    "s3x": StorageConfig(
        ml_read_median=22.0,
        ml_read_sigma=0.5,
        ml_write_median=21.0,
        ml_write_sigma=0.5,
        mf_read_median=22.0,
        mf_read_sigma=0.5,
        mf_write_median=21.0,
        mf_write_sigma=0.5,
        put_base_latency=10.0,
        put_per_mib=10.0,
        put_sigma=0.3,
        min_latency=10.0,
    ),

    # S3 Standard
    # Source: YCSB June 2025
    "s3": StorageConfig(
        ml_read_median=61.0,
        ml_read_sigma=0.5,
        ml_write_median=63.0,
        ml_write_sigma=0.5,
        mf_read_median=61.0,
        mf_read_sigma=0.5,
        mf_write_median=63.0,
        mf_write_sigma=0.5,
        put_base_latency=30.0,
        put_per_mib=20.0,
        put_sigma=0.3,
        min_latency=43.0,
    ),

    # Azure Premium Blob
    # Source: YCSB estimate
    "azurex": StorageConfig(
        ml_read_median=64.0,
        ml_read_sigma=0.5,
        ml_write_median=70.0,
        ml_write_sigma=0.5,
        mf_read_median=64.0,
        mf_read_sigma=0.5,
        mf_write_median=70.0,
        mf_write_sigma=0.5,
        put_base_latency=30.0,
        put_per_mib=15.0,
        put_sigma=0.3,
        min_latency=40.0,
    ),

    # Azure Standard Blob
    # Source: YCSB June 2025
    "azure": StorageConfig(
        ml_read_median=87.0,
        ml_read_sigma=0.5,
        ml_write_median=95.0,
        ml_write_sigma=0.5,
        mf_read_median=87.0,
        mf_read_sigma=0.5,
        mf_write_median=95.0,
        mf_write_sigma=0.5,
        put_base_latency=50.0,
        put_per_mib=25.0,
        put_sigma=0.3,
        min_latency=51.0,
    ),

    # GCS
    # Source: YCSB/Durner
    "gcp": StorageConfig(
        ml_read_median=170.0,
        ml_read_sigma=0.5,
        ml_write_median=180.0,
        ml_write_sigma=0.5,
        mf_read_median=170.0,
        mf_read_sigma=0.5,
        mf_write_median=180.0,
        mf_write_sigma=0.5,
        put_base_latency=40.0,
        put_per_mib=17.0,
        put_sigma=0.3,
        min_latency=118.0,
    ),
}

# Legacy alias: "aws" maps to "s3x" for backward compatibility
STORAGE_CONFIGS["aws"] = STORAGE_CONFIGS["s3x"]


class StorageProvider(ABC):
    """Abstract interface for storage operations.

    Handles manifest list, manifest file, and data file I/O.
    Separate from catalog operations (CAS on table pointer).

    All read/write methods accept a size_kib parameter. Even when sampling
    from a distribution, this allows future refinement where larger files
    have higher latency (base + size-dependent component).
    """

    @abstractmethod
    def get_manifest_list_read_latency(self, size_kib: float = 10.0) -> float:
        """Generate manifest list read latency in milliseconds.

        Args:
            size_kib: Manifest list size in KiB. Typical range: 1-100 KiB.
                     Grows with number of manifest file entries.
        """
        pass

    @abstractmethod
    def get_manifest_list_write_latency(self, size_kib: float = 10.0) -> float:
        """Generate manifest list write latency in milliseconds.

        Args:
            size_kib: Manifest list size in KiB.
        """
        pass

    @abstractmethod
    def get_manifest_file_read_latency(self, size_kib: float = 100.0) -> float:
        """Generate manifest file read latency in milliseconds.

        Args:
            size_kib: Manifest file size in KiB. Typical range: 10-1000 KiB.
                     Grows with number of data file entries.
        """
        pass

    @abstractmethod
    def get_manifest_file_write_latency(self, size_kib: float = 100.0) -> float:
        """Generate manifest file write latency in milliseconds.

        Args:
            size_kib: Manifest file size in KiB.
        """
        pass

    @abstractmethod
    def get_put_latency(self, size_mib: float = 1.0) -> float:
        """Generate PUT latency for a data file.

        Args:
            size_mib: File size in MiB. Data files are typically 100-500 MiB.
        """
        pass

    @abstractmethod
    def get_min_latency(self) -> float:
        """Minimum possible latency (network floor)."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this storage provider."""
        pass


class ObjectStorageProvider(StorageProvider):
    """Storage provider backed by object storage (S3, Azure, GCS)."""

    def __init__(self, provider: str):
        """Initialize with a provider name.

        Args:
            provider: One of 'instant', 's3x', 's3', 'azurex', 'azure', 'gcp'
        """
        if provider not in STORAGE_CONFIGS:
            raise ValueError(f"Unknown storage provider: {provider}. "
                           f"Valid options: {list(STORAGE_CONFIGS.keys())}")
        self._provider = provider
        self._config = STORAGE_CONFIGS[provider]

    def _sample_lognormal(self, median: float, sigma: float) -> float:
        """Sample from log-normal distribution with floor."""
        latency = np.random.lognormal(mean=np.log(median), sigma=sigma)
        return max(self._config.min_latency, latency)

    def _sample_with_size(
        self,
        median: float,
        sigma: float,
        size_kib: float,
        base_size_kib: float = 10.0
    ) -> float:
        """Sample latency with optional size scaling.

        Currently uses the median directly. Future: scale based on size.
        The size parameter is accepted for API compatibility.

        Args:
            median: Base median latency
            sigma: Log-normal sigma
            size_kib: Actual file size
            base_size_kib: Reference size the median was measured at
        """
        # TODO: Could scale median based on size ratio
        # scaled_median = median * (size_kib / base_size_kib) ** 0.3
        return self._sample_lognormal(median, sigma)

    def get_manifest_list_read_latency(self, size_kib: float = 10.0) -> float:
        return self._sample_with_size(
            self._config.ml_read_median,
            self._config.ml_read_sigma,
            size_kib,
            base_size_kib=10.0
        )

    def get_manifest_list_write_latency(self, size_kib: float = 10.0) -> float:
        return self._sample_with_size(
            self._config.ml_write_median,
            self._config.ml_write_sigma,
            size_kib,
            base_size_kib=10.0
        )

    def get_manifest_file_read_latency(self, size_kib: float = 100.0) -> float:
        return self._sample_with_size(
            self._config.mf_read_median,
            self._config.mf_read_sigma,
            size_kib,
            base_size_kib=100.0
        )

    def get_manifest_file_write_latency(self, size_kib: float = 100.0) -> float:
        return self._sample_with_size(
            self._config.mf_write_median,
            self._config.mf_write_sigma,
            size_kib,
            base_size_kib=100.0
        )

    def get_put_latency(self, size_mib: float = 1.0) -> float:
        """PUT latency scales with file size."""
        base = self._config.put_base_latency
        size_component = size_mib * self._config.put_per_mib
        median = base + size_component
        latency = np.random.lognormal(
            mean=np.log(median),
            sigma=self._config.put_sigma
        )
        return max(self._config.min_latency, latency)

    def get_min_latency(self) -> float:
        return self._config.min_latency

    @property
    def name(self) -> str:
        return self._provider


def create_storage_provider(provider: str) -> StorageProvider:
    """Factory function to create a storage provider.

    Args:
        provider: Provider name ('instant', 's3x', 's3', 'azurex', 'azure', 'gcp')

    Returns:
        StorageProvider instance
    """
    return ObjectStorageProvider(provider)
