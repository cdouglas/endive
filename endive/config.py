"""Configuration parsing and validation for the Endive simulator.

This module contains:
- Provider profiles with latency characteristics
- Latency configuration parsing functions
- Configuration validation
- Experiment hash computation

Note: Global state variables are kept in main.py to avoid import binding issues.
The configure_from_toml() function that sets globals remains in main.py.
"""

import hashlib
import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def lognormal_mu_from_median(median_ms: float) -> float:
    """Convert median to lognormal mu parameter.

    For lognormal distribution, median = exp(mu), so mu = ln(median).
    """
    return np.log(median_ms)


def convert_mean_stddev_to_lognormal(mean: float, stddev: float) -> tuple[float, float]:
    """Convert mean/stddev (normal) to approximate lognormal parameters.

    This provides backward compatibility for legacy configs.
    Uses the approximation: mu = ln(mean) - sigma²/2
    where sigma is estimated from CV (coefficient of variation).
    """
    if mean <= 0:
        return (0.0, 0.1)  # Fallback for invalid input

    # Estimate sigma from coefficient of variation
    cv = stddev / mean if mean > 0 else 0.1
    # For lognormal, CV = sqrt(exp(sigma²) - 1)
    # Solving: sigma = sqrt(ln(CV² + 1))
    sigma = np.sqrt(np.log(cv ** 2 + 1)) if cv > 0 else 0.1
    sigma = max(0.1, min(sigma, 1.5))  # Clamp to reasonable range

    # mu = ln(median) and for lognormal, median = mean / exp(sigma²/2)
    mu = np.log(mean) - (sigma ** 2 / 2)

    return mu, sigma


# =============================================================================
# Provider Profiles - Based on June 2025 YCSB benchmark measurements
# =============================================================================

# Provider profiles based on YCSB benchmark measurements (June 2025)
# Source: analysis/simulation_summary.md and analysis/distributions.json
#
# IMPORTANT: Storage tiers have distinct latency characteristics.
# Do NOT merge tiers (e.g., S3 vs S3 Express, Azure Blob vs Azure Premium).
#
# Parameters:
#   median: Median latency in ms (used to compute mu = ln(median))
#   sigma: Lognormal sigma parameter (controls tail heaviness)
#   failure: Separate distribution for failed operations (optional)
#     - If present, use these parameters instead of success * multiplier
#     - failure.median: Median latency for failed operations
#     - failure.sigma: Sigma for failed operations
#   failure_multiplier: Legacy - ratio of fail/success latency (used if no failure dist)
#   contention_scaling: Latency multiplier at 16 threads vs 1 thread
PROVIDER_PROFILES = {
    # AWS S3 Standard - CAS only (no conditional append support)
    # CAS: Success median=60.8ms, sigma=0.14 (YCSB June 2025)
    # Failure: median=65ms (1.07x), similar distribution shape
    # PUT latency model from Durner et al. VLDB 2023:
    #   latency = base_latency + size * latency_per_mib
    #   base_latency ~30ms (first byte), throughput ~50 MiB/s (20 ms/MiB)
    "s3": {
        "put": {
            "base_latency_ms": 30,       # Durner et al. Fig 2: ~30ms first byte
            "latency_per_mib_ms": 20,    # Durner et al. Sec 2.8: 50 MiB/s = 20ms/MiB
            "sigma": 0.30,               # Variance estimate from Fig 3 (25-95 MiB/s range)
            "_provenance": "Durner et al. VLDB 2023, Section 2.3, 2.8"
        },
        "manifest_list": {
            "read": {"median": 61, "sigma": 0.14},
            "write": {"median": 63, "sigma": 0.14}
        },
        "manifest_file": {
            "read": {"median": 61, "sigma": 0.14},
            "write": {"median": 63, "sigma": 0.14}
        },
        "cas": {
            "median": 61, "sigma": 0.14,
            "failure": {"median": 65, "sigma": 0.38},  # Heavier tail on failures
            "failure_multiplier": 1.22  # Legacy fallback
        },
        "append": None,  # S3 Standard does not support conditional append
        "contention_scaling": {"cas": 0.97, "append": None},
    },
    # AWS S3 Express One Zone - CAS + conditional append
    # CAS success: median=22.4ms, sigma=0.22 (YCSB June 2025)
    # CAS failure: median=21.1ms (0.95x), similar shape
    # Append success: median=20.5ms, sigma=0.25
    # Append failure: median=22.6ms (1.09x), similar shape
    # PUT latency: Estimated from S3 Express positioning as ~3x faster than S3 Standard
    #   S3 Express is designed for single-digit ms latency for small objects
    "s3x": {
        "put": {
            "base_latency_ms": 10,       # Estimate: ~3x faster than S3 Standard (30ms)
            "latency_per_mib_ms": 10,    # Estimate: ~2x faster throughput (100 MiB/s)
            "sigma": 0.25,               # Similar variance to CAS operations
            "_provenance": "Estimate based on S3 Express positioning vs S3 Standard (Durner et al.)"
        },
        "manifest_list": {
            "read": {"median": 22, "sigma": 0.22},
            "write": {"median": 21, "sigma": 0.25}
        },
        "manifest_file": {
            "read": {"median": 22, "sigma": 0.22},
            "write": {"median": 21, "sigma": 0.25}
        },
        "cas": {
            "median": 22, "sigma": 0.22,
            "failure": {"median": 21, "sigma": 0.22},  # Failures slightly faster
            "failure_multiplier": 0.95
        },
        "append": {
            "median": 21, "sigma": 0.25,
            "failure": {"median": 23, "sigma": 0.25},
            "failure_multiplier": 1.09
        },
        "contention_scaling": {"cas": 1.77, "append": 1.85},
    },
    # Azure Blob Storage (Standard tier) - CAS + conditional append
    # CAS success: median=93ms, sigma=0.82 (very heavy tails) (YCSB June 2025)
    # CAS failure: median=75ms (0.81x), failures actually faster
    # Append success: median=87ms, sigma=0.28
    # Append failure: median=2072ms, sigma=0.68 (COMPLETELY DIFFERENT DISTRIBUTION!)
    # PUT latency: Estimated from Azure documentation and Durner et al. Cloud X/Y comparison
    #   Azure typically has higher latency than AWS; Cloud Y in Durner had 12-15 ms/MiB
    "azure": {
        "put": {
            "base_latency_ms": 50,       # Estimate: Azure typically higher latency than AWS
            "latency_per_mib_ms": 25,    # Estimate: ~40 MiB/s, slower than S3
            "sigma": 0.50,               # Higher variance due to heavy tails in CAS
            "_provenance": "Estimate based on Azure vs AWS positioning; Durner et al. Cloud Y data"
        },
        "manifest_list": {
            "read": {"median": 87, "sigma": 0.28},
            "write": {"median": 95, "sigma": 0.28}
        },
        "manifest_file": {
            "read": {"median": 87, "sigma": 0.28},
            "write": {"median": 95, "sigma": 0.28}
        },
        "cas": {
            "median": 93, "sigma": 0.82,
            "failure": {"median": 75, "sigma": 0.90},  # Failures faster but heavier tail
            "failure_multiplier": 0.81
        },
        "append": {
            "median": 87, "sigma": 0.28,
            # CRITICAL: Append failures have completely different distribution!
            # Failure median=2072ms, p99/p50≈4.9 -> sigma≈0.68
            "failure": {"median": 2072, "sigma": 0.68},
            "failure_multiplier": 31.6
        },
        "contention_scaling": {"cas": 5.4, "append": 1.07},
    },
    # Azure Premium Block Blob - CAS + conditional append
    # CAS success: median=64ms, sigma=0.73 (YCSB June 2025)
    # CAS failure: median=82ms (1.28x)
    # Append success: median=70ms, sigma=0.23
    # Append failure: median=2534ms, sigma=0.65 (COMPLETELY DIFFERENT DISTRIBUTION!)
    # PUT latency: Estimated from Premium tier positioning (~1.5x faster than Standard)
    "azurex": {
        "put": {
            "base_latency_ms": 30,       # Estimate: Premium ~1.7x faster than Standard
            "latency_per_mib_ms": 15,    # Estimate: ~67 MiB/s, faster than Standard
            "sigma": 0.40,               # Lower variance than Standard tier
            "_provenance": "Estimate based on Azure Premium vs Standard tier ratio"
        },
        "manifest_list": {
            "read": {"median": 70, "sigma": 0.23},
            "write": {"median": 72, "sigma": 0.23}
        },
        "manifest_file": {
            "read": {"median": 70, "sigma": 0.23},
            "write": {"median": 72, "sigma": 0.23}
        },
        "cas": {
            "median": 64, "sigma": 0.73,
            "failure": {"median": 82, "sigma": 0.85},
            "failure_multiplier": 1.28
        },
        "append": {
            "median": 70, "sigma": 0.23,
            # CRITICAL: Append failures have completely different distribution!
            "failure": {"median": 2534, "sigma": 0.65},
            "failure_multiplier": 36.2
        },
        "contention_scaling": {"cas": 6.0, "append": 1.02},
    },
    # GCP Cloud Storage - CAS only (no conditional append data)
    # CAS success: median=170ms, sigma=0.91 (extremely heavy tails) (YCSB June 2025)
    # CAS failure: mean=7111ms, estimated median≈4000-5000ms based on heavy tails
    # PUT latency: Durner et al. Cloud X had 12-15 ms/MiB; GCP likely similar or slower
    "gcp": {
        "put": {
            "base_latency_ms": 40,       # Estimate: similar to Cloud X in Durner et al.
            "latency_per_mib_ms": 17,    # Durner et al. Cloud X: 12-15 ms/MiB, estimate 17
            "sigma": 0.60,               # High variance based on CAS heavy tails
            "_provenance": "Estimate based on Durner et al. Cloud X data (likely GCP)"
        },
        "manifest_list": {
            "read": {"median": 170, "sigma": 0.91},
            "write": {"median": 200, "sigma": 0.91}
        },
        "manifest_file": {
            "read": {"median": 170, "sigma": 0.91},
            "write": {"median": 200, "sigma": 0.91}
        },
        "cas": {
            "median": 170, "sigma": 0.91,
            # Failures are dramatically slower (mean 7111ms vs 530ms)
            "failure": {"median": 4500, "sigma": 1.0},  # Estimated from mean=7111
            "failure_multiplier": 13.4
        },
        "append": None,  # No append data available for GCP
        "contention_scaling": {"cas": 0.70, "append": None},
    },
    # Hypothetical infinitely fast system for "what if" experiments
    "instant": {
        "put": {
            "base_latency_ms": 0.5,      # Near-instant base latency
            "latency_per_mib_ms": 0.1,   # ~10 GB/s effective throughput
            "sigma": 0.1,
            "_provenance": "Synthetic: infinitely fast for baseline comparison"
        },
        "manifest_list": {
            "read": {"median": 1, "sigma": 0.1},
            "write": {"median": 1, "sigma": 0.1}
        },
        "manifest_file": {
            "read": {"median": 1, "sigma": 0.1},
            "write": {"median": 1, "sigma": 0.1}
        },
        "cas": {"median": 1, "sigma": 0.1, "failure_multiplier": 1.0},
        "append": {"median": 1, "sigma": 0.1, "failure_multiplier": 1.0},
        "contention_scaling": {"cas": 1.0, "append": 1.0},
    },
}

# Backward compatibility aliases
# "aws" was previously used but mixed S3 and S3 Express data
# Map to S3 Express since it supports both CAS and append
PROVIDER_PROFILES["aws"] = PROVIDER_PROFILES["s3x"]


def parse_latency_config(config_section: dict, defaults: dict = None) -> dict:
    """Parse latency configuration supporting both legacy and new formats.

    Legacy format (normal distribution):
        {'mean': 50.0, 'stddev': 5.0}

    New format (lognormal distribution):
        {'median': 50.0, 'sigma': 0.3}
        {'median': 50.0, 'sigma': 0.3, 'failure_multiplier': 1.2}

    Returns dict with 'mu', 'sigma', 'distribution', and optionally 'failure_multiplier'.
    """
    result = {}

    # Check for new lognormal format
    if 'median' in config_section:
        median = config_section['median']
        sigma = config_section.get('sigma', 0.3)
        result['mu'] = lognormal_mu_from_median(median)
        result['sigma'] = sigma
        result['distribution'] = 'lognormal'
        if 'failure_multiplier' in config_section:
            result['failure_multiplier'] = config_section['failure_multiplier']
    # Legacy normal format
    elif 'mean' in config_section:
        mean = config_section['mean']
        stddev = config_section.get('stddev', mean * 0.1)
        # Check if we should convert to lognormal
        if config_section.get('distribution', 'normal') == 'lognormal':
            mu, sigma = convert_mean_stddev_to_lognormal(mean, stddev)
            result['mu'] = mu
            result['sigma'] = sigma
            result['distribution'] = 'lognormal'
        else:
            # Keep as normal distribution (legacy behavior)
            result['mean'] = mean
            result['stddev'] = stddev
            result['distribution'] = 'normal'
    # Use defaults if provided
    elif defaults:
        result = defaults.copy()
    else:
        # Fallback defaults
        result = {'mu': np.log(50), 'sigma': 0.3, 'distribution': 'lognormal'}

    return result


def get_provider_latency_config(provider: str, operation: str, sub_op: str = None) -> dict | None:
    """Get latency configuration from a provider profile.

    Args:
        provider: Provider name (aws, azure, gcp, instant)
        operation: Operation type (cas, append, manifest_list, manifest_file)
        sub_op: Sub-operation for nested configs (read, write)

    Returns:
        Dict with mu, sigma, distribution, and optionally failure_multiplier.
        Returns None if provider or operation not found.
    """
    if provider not in PROVIDER_PROFILES:
        return None

    profile = PROVIDER_PROFILES[provider]
    if operation not in profile:
        return None

    op_config = profile[operation]
    if op_config is None:
        return None

    # Handle nested configs (manifest_list, manifest_file)
    if sub_op:
        if sub_op not in op_config:
            return None
        op_config = op_config[sub_op]

    # Convert profile format to internal format
    result = {
        'mu': lognormal_mu_from_median(op_config['median']),
        'sigma': op_config['sigma'],
        'distribution': 'lognormal'
    }
    if 'failure_multiplier' in op_config:
        result['failure_multiplier'] = op_config['failure_multiplier']

    # Include separate failure distribution if present
    if 'failure' in op_config:
        result['failure'] = {
            'median': op_config['failure']['median'],
            'sigma': op_config['failure']['sigma']
        }

    return result


def apply_provider_defaults(provider: str, storage_cfg: dict, catalog_backend: str, catalog_cfg: dict) -> dict:
    """Apply provider profile defaults, then overlay explicit config.

    Args:
        provider: Provider name or None for custom config
        storage_cfg: Explicit storage configuration from TOML
        catalog_backend: "storage", "service", or "fifo_queue"
        catalog_cfg: Catalog configuration section (may have service/fifo_queue subsections)

    Returns:
        Dict with resolved latency configurations for all operations.
    """
    result = {}

    # Start with provider defaults if specified
    if provider and provider in PROVIDER_PROFILES:
        profile = PROVIDER_PROFILES[provider]

        # Storage operations (always from storage provider)
        for op in ['manifest_list', 'manifest_file']:
            if profile.get(op):
                result[f'T_{op.upper()}'] = {
                    'read': get_provider_latency_config(provider, op, 'read'),
                    'write': get_provider_latency_config(provider, op, 'write')
                }

        # Catalog operations depend on backend
        if catalog_backend == 'storage':
            # Catalog uses storage provider latencies
            if profile.get('cas'):
                result['T_CAS'] = get_provider_latency_config(provider, 'cas')
            if profile.get('append'):
                result['T_APPEND'] = get_provider_latency_config(provider, 'append')
        elif catalog_backend == 'service':
            # Check for catalog.service.provider or explicit config
            service_cfg = catalog_cfg.get('service', {})
            service_provider = service_cfg.get('provider', provider)
            if service_provider and service_provider in PROVIDER_PROFILES:
                service_profile = PROVIDER_PROFILES[service_provider]
                if service_profile.get('cas'):
                    result['T_CAS'] = get_provider_latency_config(service_provider, 'cas')
                if service_profile.get('append'):
                    result['T_APPEND'] = get_provider_latency_config(service_provider, 'append')
        elif catalog_backend == 'fifo_queue':
            # FIFO queue has its own config, checkpoints use storage
            queue_cfg = catalog_cfg.get('fifo_queue', {})
            if 'append' in queue_cfg:
                result['T_APPEND'] = parse_latency_config(queue_cfg['append'])
            elif provider and PROVIDER_PROFILES.get(provider, {}).get('append'):
                # Fallback to provider append config
                result['T_APPEND'] = get_provider_latency_config(provider, 'append')

        # Size-based PUT latency (always from storage provider)
        if profile.get('put'):
            result['T_PUT'] = profile['put']

    return result


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
    num_groups = catalog.get('num_groups', 1)
    if num_tables <= 0:
        errors.append("catalog.num_tables must be > 0")
    if num_groups <= 0:
        errors.append("catalog.num_groups must be > 0")
    if num_groups > num_tables:
        errors.append(f"catalog.num_groups ({num_groups}) cannot exceed num_tables ({num_tables})")

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

        # Warn about complex interactions
        if num_groups > 1:
            warnings.append(f"partition.enabled=true with num_groups={num_groups} > 1 creates nested isolation (table groups + partitions); this is valid but may be unintended")

    return errors, warnings
