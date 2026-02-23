"""Configuration parsing and validation for the Endive simulator.

This module contains:
- load_simulation_config(): unified entry point per SPEC.md §7
- Configuration validation
- Experiment hash computation

Provider latency profiles are in endive/providers/*.toml, loaded by
endive.storage._load_provider_profile().
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tomllib

from endive.catalog import CASCatalog, InstantCatalog
from endive.conflict_detector import (
    PartitionOverlapConflictDetector,
    ProbabilisticConflictDetector,
)
from endive.simulation import SimulationConfig
from endive.storage import (
    FixedLatency,
    LognormalLatency,
    create_provider,
)
from endive.workload import Workload, WorkloadConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration error
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Fatal configuration error(s)."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


# ---------------------------------------------------------------------------
# Unified config loader (SPEC.md §7)
# ---------------------------------------------------------------------------

def load_simulation_config(
    config_path: str,
    *,
    seed_override: int | None = None,
) -> SimulationConfig:
    """Load simulation configuration from TOML file.

    This is the ONLY entry point for configuration.
    All parameters are validated before returning.

    Args:
        config_path: Path to TOML configuration file.
        seed_override: If provided, overrides the seed in the config file.

    Returns:
        Fully constructed SimulationConfig ready to run.
    """
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    # Apply seed override
    if seed_override is not None:
        raw.setdefault("simulation", {})["seed"] = seed_override

    # Validate configuration
    errors, warnings = validate_config(raw)
    if errors:
        raise ConfigurationError(errors)
    for warning in warnings:
        logger.warning(warning)

    # Extract shared topology
    catalog_cfg = raw.get("catalog", {})
    num_tables = catalog_cfg.get("num_tables", 1)
    partitions_per_table = _build_partition_counts(catalog_cfg, num_tables)

    # Build seed-derived RNG
    sim_cfg = raw.get("simulation", {})
    seed = sim_cfg.get("seed")
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # Build components in dependency order
    storage = _build_storage_provider(raw.get("storage", {}), rng)
    catalog = _build_catalog(catalog_cfg, storage, num_tables, partitions_per_table)
    workload = _build_workload(raw.get("transaction", {}), num_tables,
                                partitions_per_table, seed)
    conflict_detector = _build_conflict_detector(
        raw.get("transaction", {}), raw.get("partition", {}), rng,
    )

    # Extract simulation parameters
    txn_cfg = raw.get("transaction", {})
    ml_mode = txn_cfg.get("manifest_list_mode", "rewrite")

    return SimulationConfig(
        duration_ms=sim_cfg.get("duration_ms", 3600000),
        seed=seed,
        storage_provider=storage,
        catalog=catalog,
        workload=workload,
        conflict_detector=conflict_detector,
        max_retries=txn_cfg.get("retry", 10),
        ml_append_mode=(ml_mode == "append"),
    )


def _build_partition_counts(
    catalog_cfg: dict,
    num_tables: int,
) -> Tuple[int, ...]:
    """Build per-table partition counts from config.

    Supports:
    - catalog.partition.num_partitions (uniform for all tables)
    - catalog.partition.per_table (explicit per-table list)
    - Default: 1 partition per table
    """
    partition_cfg = catalog_cfg.get("partition", {})
    if not partition_cfg:
        return tuple([1] * num_tables)

    if "per_table" in partition_cfg:
        counts = tuple(partition_cfg["per_table"])
        if len(counts) != num_tables:
            raise ConfigurationError([
                f"catalog.partition.per_table has {len(counts)} entries "
                f"but num_tables={num_tables}"
            ])
        return counts

    n = partition_cfg.get("num_partitions", 1)
    return tuple([n] * num_tables)


def _build_storage_provider(storage_cfg: dict, rng: np.random.RandomState):
    """Build StorageProvider from [storage] config section."""
    provider_name = storage_cfg.get("provider", "instant")
    return create_provider(provider_name, rng=rng)


def _build_catalog(
    catalog_cfg: dict,
    storage,
    num_tables: int,
    partitions_per_table: Tuple[int, ...],
):
    """Build Catalog from [catalog] config section."""
    mode = catalog_cfg.get("mode", "cas")
    backend = catalog_cfg.get("backend", "storage")

    # Handle catalog.backend = "service" with [catalog.service] section
    if backend == "service":
        service_cfg = catalog_cfg.get("service", {})
        service_provider = service_cfg.get("provider", "")
        if service_provider == "instant":
            latency = service_cfg.get("latency_ms", 1.0)
            return InstantCatalog(
                num_tables=num_tables,
                partitions_per_table=partitions_per_table,
                latency_ms=latency,
            )
        # Service backend with non-instant provider — use CAS with storage
        return CASCatalog(
            storage=storage,
            num_tables=num_tables,
            partitions_per_table=partitions_per_table,
        )

    if mode == "cas" and catalog_cfg.get("provider", "") == "instant":
        # Explicit instant catalog
        latency = catalog_cfg.get("latency_ms", 1.0)
        return InstantCatalog(
            num_tables=num_tables,
            partitions_per_table=partitions_per_table,
            latency_ms=latency,
        )

    # Check if storage provider is instant → use InstantCatalog
    from endive.storage import InstantStorageProvider
    if isinstance(storage, InstantStorageProvider):
        latency = catalog_cfg.get("latency_ms", 1.0)
        return InstantCatalog(
            num_tables=num_tables,
            partitions_per_table=partitions_per_table,
            latency_ms=latency,
        )

    # Default: CASCatalog with storage provider
    return CASCatalog(
        storage=storage,
        num_tables=num_tables,
        partitions_per_table=partitions_per_table,
    )


def _build_inter_arrival(txn_cfg: dict):
    """Build inter-arrival LatencyDistribution from config."""
    ia_cfg = txn_cfg.get("inter_arrival", {})
    dist = ia_cfg.get("distribution", "exponential")

    if dist == "fixed":
        return FixedLatency(latency_ms=ia_cfg.get("value", 100.0))
    elif dist == "exponential":
        # For exponential inter-arrival, use lognormal approximation
        # scale is the mean inter-arrival time
        scale = ia_cfg.get("scale", 100.0)
        sigma = ia_cfg.get("sigma", 0.5)
        return LognormalLatency.from_median(median_ms=scale, sigma=sigma)
    else:
        # Fallback: use scale as median
        scale = ia_cfg.get("scale", ia_cfg.get("mean", 100.0))
        sigma = ia_cfg.get("sigma", 0.5)
        return LognormalLatency.from_median(median_ms=scale, sigma=sigma)


def _build_runtime(txn_cfg: dict):
    """Build runtime LatencyDistribution from config."""
    rt_cfg = txn_cfg.get("runtime", {})
    mean = rt_cfg.get("mean", 180000)
    sigma = rt_cfg.get("sigma", 1.5)
    min_val = rt_cfg.get("min", 1.0)
    return LognormalLatency.from_median(
        median_ms=mean, sigma=sigma, min_latency_ms=min_val,
    )


def _build_workload(
    txn_cfg: dict,
    num_tables: int,
    partitions_per_table: Tuple[int, ...],
    seed: int | None,
):
    """Build Workload from [transaction] config section."""
    inter_arrival = _build_inter_arrival(txn_cfg)
    runtime = _build_runtime(txn_cfg)

    # Operation type weights
    op_types = txn_cfg.get("operation_types", {})
    fa_weight = op_types.get("fast_append", 1.0)
    ma_weight = op_types.get("merge_append", 0.0)
    vo_weight = op_types.get("validated_overwrite", 0.0)

    # MergeAppend parameters
    cm = txn_cfg.get("conflicting_manifests", {})
    manifests_per_commit = cm.get("mean", 1.5)

    wl_config = WorkloadConfig(
        inter_arrival=inter_arrival,
        runtime=runtime,
        num_tables=num_tables,
        partitions_per_table=partitions_per_table,
        fast_append_weight=fa_weight,
        merge_append_weight=ma_weight,
        validated_overwrite_weight=vo_weight,
        manifests_per_concurrent_commit=manifests_per_commit,
    )

    # Workload seed is derived from simulation seed
    wl_seed = (seed + 100) if seed is not None else None
    return Workload(wl_config, seed=wl_seed)


def _build_conflict_detector(
    txn_cfg: dict,
    partition_cfg: dict,
    rng: np.random.RandomState,
):
    """Build ConflictDetector from config."""
    prob = txn_cfg.get("real_conflict_probability", 0.0)

    # If partitions are enabled and probability-based detection is not forced,
    # use partition overlap detector
    if partition_cfg.get("enabled", False):
        return PartitionOverlapConflictDetector()

    return ProbabilisticConflictDetector(prob, rng=rng)




def compute_experiment_hash(config: dict) -> str:
    """Compute deterministic hash of config + simulator code.

    The hash includes:
    1. All config parameters except 'seed' and 'experiment.label'
    2. Hash of simulator code files (endive/*.py)

    This ensures that:
    - Same config → same hash (reproducible)
    - Different seeds → same hash (seeds go in different dirs)
    - Code changes → different hash (invalidates old results)

    Returns:
        8-character hex hash string
    """
    # 1. Hash configuration (excluding seed and label)
    config_for_hash = dict(config)

    # Remove seed from simulation section
    if 'simulation' in config_for_hash and 'seed' in config_for_hash['simulation']:
        config_for_hash = dict(config_for_hash)  # Copy
        config_for_hash['simulation'] = dict(config_for_hash['simulation'])
        del config_for_hash['simulation']['seed']

    # Remove experiment section entirely (contains label)
    if 'experiment' in config_for_hash:
        config_for_hash = dict(config_for_hash)
        del config_for_hash['experiment']

    # Serialize config deterministically
    config_str = json.dumps(config_for_hash, sort_keys=True)

    # 2. Hash simulator code files
    code_hash = hashlib.sha256()
    endive_dir = Path(__file__).parent

    # Include all .py files in endive/ directory
    for py_file in sorted(endive_dir.glob("*.py")):
        with open(py_file, 'rb') as f:
            code_hash.update(f.read())

    # Include provider TOML files (latency config)
    providers_dir = endive_dir / "providers"
    if providers_dir.is_dir():
        for toml_file in sorted(providers_dir.glob("*.toml")):
            with open(toml_file, 'rb') as f:
                code_hash.update(f.read())

    # 3. Combine config and code hashes
    combined = hashlib.sha256()
    combined.update(config_str.encode('utf-8'))
    combined.update(code_hash.digest())

    # Return first 8 characters of hex digest
    return combined.hexdigest()[:8]


def validate_config(config: dict) -> tuple[list[str], list[str]]:
    """Validate configuration and return errors/warnings.

    Returns:
        (errors, warnings) where:
        - errors: List of fatal configuration errors
        - warnings: List of non-fatal warnings
    """
    errors = []
    warnings = []

    # Catalog section
    catalog = config.get('catalog', {})
    num_tables = catalog.get('num_tables', 1)
    if num_tables <= 0:
        errors.append("catalog.num_tables must be > 0")

    # Validate catalog mode
    catalog_mode = catalog.get('mode', 'cas')
    if catalog_mode not in ['cas', 'append']:
        errors.append(f"catalog.mode must be 'cas' or 'append', got '{catalog_mode}'")

    # Transaction section (optional for partition-only validation)
    txn = config.get('transaction', {})

    # Inter-arrival distribution (only validate if transaction section exists)
    if 'transaction' in config and 'inter_arrival' in txn:
        inter_arrival = txn.get('inter_arrival', {})
        dist = inter_arrival.get('distribution', 'exponential')
        valid_dists = ['fixed', 'exponential', 'uniform', 'normal', 'poisson']
        if dist not in valid_dists:
            errors.append(f"transaction.inter_arrival.distribution must be one of {valid_dists}, got '{dist}'")

        # Distribution-specific parameter validation
        if dist == 'fixed':
            if 'value' not in inter_arrival:
                errors.append("transaction.inter_arrival.value required for fixed distribution")
        elif dist == 'exponential':
            if 'scale' not in inter_arrival:
                errors.append("transaction.inter_arrival.scale required for exponential distribution")
            elif inter_arrival['scale'] <= 0:
                errors.append(f"transaction.inter_arrival.scale must be > 0, got {inter_arrival['scale']}")
        elif dist == 'uniform':
            min_val = inter_arrival.get('min', 0)
            max_val = inter_arrival.get('max', 0)
            if min_val >= max_val:
                errors.append(f"transaction.inter_arrival.min ({min_val}) must be < max ({max_val})")
        elif dist == 'normal':
            if 'mean' not in inter_arrival:
                errors.append("transaction.inter_arrival.mean required for normal distribution")
            if 'stddev' not in inter_arrival:
                errors.append("transaction.inter_arrival.stddev required for normal distribution")
        elif dist == 'poisson':
            if 'rate' not in inter_arrival:
                errors.append("transaction.inter_arrival.rate required for poisson distribution")

    # Runtime configuration
    runtime = txn.get('runtime', {})
    if runtime:
        mean = runtime.get('mean')
        if mean is not None and mean <= 0:
            errors.append(f"transaction.runtime.mean must be > 0, got {mean}")
        min_runtime = runtime.get('min', 0)
        if min_runtime < 0:
            errors.append(f"transaction.runtime.min must be >= 0, got {min_runtime}")
        if mean is not None and min_runtime > mean:
            warnings.append(f"transaction.runtime.min ({min_runtime}) > mean ({mean}); transactions will cluster at minimum")

    # Conflict probability
    real_conflict_prob = txn.get('real_conflict_probability', 0.0)
    if not 0.0 <= real_conflict_prob <= 1.0:
        errors.append(f"transaction.real_conflict_probability must be in [0, 1], got {real_conflict_prob}")

    # Operation type weights validation
    op_types = txn.get('operation_types', {})
    if op_types:
        valid_op_types = {'fast_append', 'merge_append', 'validated_overwrite'}
        for op_name in op_types.keys():
            if op_name not in valid_op_types:
                errors.append(f"Unknown operation type: '{op_name}'. Valid types: {valid_op_types}")

        # Check weights are non-negative
        for op_name, weight in op_types.items():
            if not isinstance(weight, (int, float)):
                errors.append(f"operation_types.{op_name} must be a number, got {type(weight).__name__}")
            elif weight < 0:
                errors.append(f"operation_types.{op_name} must be >= 0, got {weight}")

        # Warn if weights don't sum to 1.0 (will be normalized)
        weight_sum = sum(w for w in op_types.values() if isinstance(w, (int, float)))
        if weight_sum > 0 and abs(weight_sum - 1.0) > 0.01:
            warnings.append(f"operation_types weights sum to {weight_sum:.3f}, will be normalized to 1.0")

        # Warn about validated_overwrite with real_conflict_probability > 0
        validated_weight = op_types.get('validated_overwrite', 0.0)
        if validated_weight > 0 and real_conflict_prob > 0:
            warnings.append(
                f"validated_overwrite ({validated_weight*100:.0f}%) with real_conflict_probability={real_conflict_prob:.1%} "
                f"will cause ~{validated_weight * real_conflict_prob * 100:.1f}% of transactions to abort (ValidationException)"
            )

    # Storage section (optional for partition-only validation)
    storage = config.get('storage', {})
    provider = storage.get('provider')
    valid_providers = ['s3', 's3x', 'azure', 'azurex', 'gcp', 'instant', 'aws']
    if provider and provider not in valid_providers:
        errors.append(f"storage.provider must be one of {valid_providers}, got '{provider}'")

    # Manifest list mode
    ml_mode = txn.get('manifest_list_mode', 'rewrite')
    if ml_mode not in ['rewrite', 'append']:
        errors.append(f"transaction.manifest_list_mode must be 'rewrite' or 'append', got '{ml_mode}'")

    # Warn about append mode without append support
    if provider in ['s3', 'gcp'] and catalog_mode == 'append':
        warnings.append(f"catalog.mode='append' with provider='{provider}' which doesn't support conditional append; using CAS semantics")

    # Partition configuration
    partition = config.get('partition', {})
    if partition.get('enabled', False):
        n_partitions = partition.get('num_partitions', 10)
        if n_partitions <= 0:
            errors.append(f"partition.num_partitions must be > 0, got {n_partitions}")

        selection = partition.get('selection', {})
        dist = selection.get('distribution', 'zipf')
        if dist not in ['zipf', 'uniform']:
            errors.append(f"partition.selection.distribution must be 'zipf' or 'uniform', got '{dist}'")

        if dist == 'zipf':
            alpha = selection.get('zipf_alpha', 1.5)
            if alpha <= 0:
                errors.append(f"partition.selection.zipf_alpha must be > 0, got {alpha}")

        parts_mean = partition.get('partitions_per_txn_mean', 3.0)
        parts_max = partition.get('partitions_per_txn_max', 10)
        if parts_mean <= 0:
            errors.append(f"partition.partitions_per_txn_mean must be > 0, got {parts_mean}")
        if parts_max <= 0:
            errors.append(f"partition.partitions_per_txn_max must be > 0, got {parts_max}")
        if parts_max > n_partitions:
            warnings.append(f"partition.partitions_per_txn_max ({parts_max}) > num_partitions ({n_partitions}); will be clamped to {n_partitions}")

        # Warn about meaningless configurations
        if n_partitions == 1:
            warnings.append("partition.num_partitions = 1 is meaningless (no partition distribution); consider disabling partition mode or increasing num_partitions")

        if parts_mean >= n_partitions:
            warnings.append(f"partition.partitions_per_txn_mean ({parts_mean}) >= num_partitions ({n_partitions}); transactions will touch all partitions, defeating partition isolation")

    return errors, warnings
