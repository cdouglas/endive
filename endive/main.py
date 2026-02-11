#!/usr/bin/env python

import argparse
import itertools
import logging
import simpy
import sys
import tomllib
import numpy as np
from tqdm import tqdm
from endive.capstats import Stats, truncated_zipf_pmf, lognormal_params_from_mean_and_sigma
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Simulation parameters
SIM_DURATION_MS: int
SIM_OUTPUT_PATH: str
SIM_SEED: int | None

# Experiment parameters
EXPERIMENT_LABEL: str | None

N_TABLES: int
N_GROUPS: int  # Number of table groups
GROUP_SIZE_DIST: str  # Distribution of group sizes
LONGTAIL_PARAMS: dict  # Parameters for longtail distribution
TABLE_TO_GROUP: dict  # Mapping from table ID to group ID
GROUP_TO_TABLES: dict  # Mapping from group ID to list of table IDs
N_TXN_RETRY: int
# Storage operation latencies
# New format: {'mu': float, 'sigma': float, 'distribution': 'lognormal'}
# Legacy format: {'mean': float, 'stddev': float, 'distribution': 'normal'}
T_CAS: dict
T_METADATA_ROOT: dict  # {'read': {...}, 'write': {...}}
T_MANIFEST_LIST: dict
T_MANIFEST_FILE: dict
# Latency distribution type: "lognormal" (recommended) or "normal" (legacy)
LATENCY_DISTRIBUTION: str
MAX_PARALLEL: int  # Maximum parallel manifest operations during conflict resolution
MIN_LATENCY: float  # Minimum latency for any storage operation (prevents unrealistic zeros)
# lognormal distribution of transaction runtimes
T_MIN_RUNTIME: int
T_RUNTIME_MU: float
T_RUNTIME_SIGMA: float
# inter-arrival distribution parameters
INTER_ARRIVAL_DIST: str
INTER_ARRIVAL_PARAMS: dict
# tables per transaction (prob. mass function)
N_TBL_PMF: float
# which tables are selected; (zipf, 0 most likely, so on)
TBL_R_PMF: float
# number of tables written (subset read)
N_TBL_W_PMF: float
# Conflict resolution parameters
REAL_CONFLICT_PROBABILITY: float  # Probability that a conflict is "real" (requires manifest file ops)
CONFLICTING_MANIFESTS_DIST: str  # Distribution of conflicting manifests
CONFLICTING_MANIFESTS_PARAMS: dict  # Parameters for conflicting manifests distribution
# Retry backoff parameters
RETRY_BACKOFF_ENABLED: bool  # Whether exponential backoff is enabled
RETRY_BACKOFF_BASE_MS: float  # Base backoff time in milliseconds
RETRY_BACKOFF_MULTIPLIER: float  # Multiplier for each retry
RETRY_BACKOFF_MAX_MS: float  # Maximum backoff time in milliseconds
RETRY_BACKOFF_JITTER: float  # Jitter factor (0.0 to 1.0) for randomization

# Storage and catalog configuration
STORAGE_PROVIDER: str | None  # "aws", "azure", "gcp", "instant", or None for custom
CATALOG_BACKEND: str  # "storage" (default), "service", or "fifo_queue"

# Append mode parameters (catalog mode = "append")
CATALOG_MODE: str  # "cas" (default) or "append"
COMPACTION_THRESHOLD: int  # Bytes before triggering compaction (default 16MB)
COMPACTION_MAX_ENTRIES: int  # Max entries before compaction (0 = disabled, use size only)
LOG_ENTRY_SIZE: int  # Average bytes per log entry (for threshold calculation)
T_APPEND: dict  # {'mean': float, 'stddev': float} - Append operation latency
T_LOG_ENTRY_READ: dict  # {'mean': float, 'stddev': float} - Per-entry log read latency
T_COMPACTION: dict  # {'mean': float, 'stddev': float} - Compaction CAS latency (larger payload)
# Table metadata configuration
TABLE_METADATA_INLINED: bool  # Whether table metadata is inlined in catalog/intention record
T_TABLE_METADATA_R: dict  # {'mean': float, 'stddev': float} - Table metadata read latency
T_TABLE_METADATA_W: dict  # {'mean': float, 'stddev': float} - Table metadata write latency
# Manifest list append mode
MANIFEST_LIST_MODE: str  # "rewrite" (default) or "append"
MANIFEST_LIST_SEAL_THRESHOLD: int  # Bytes before manifest list is sealed/rewritten (0 = disabled)
MANIFEST_LIST_ENTRY_SIZE: int  # Average bytes per manifest list entry

STATS = Stats()

# Contention tracking for latency scaling
CONTENTION_SCALING_ENABLED: bool = False  # Enable contention-based latency scaling
CATALOG_CONTENTION_SCALING: dict = {}  # {'cas': 1.4, 'append': 1.8} from provider


class ContentionTracker:
    """Track concurrent catalog operations for latency scaling.

    Based on measurements showing that latency increases with contention:
    - AWS CAS: 1.4x increase at 16 threads vs 1 thread
    - AWS Append: 1.8x increase
    - Azure CAS: 5.8x increase (scales poorly)
    """

    def __init__(self):
        self.active_cas = 0
        self.active_append = 0

    def enter_cas(self):
        """Mark start of CAS operation."""
        self.active_cas += 1

    def exit_cas(self):
        """Mark end of CAS operation."""
        self.active_cas = max(0, self.active_cas - 1)

    def enter_append(self):
        """Mark start of append operation."""
        self.active_append += 1

    def exit_append(self):
        """Mark end of append operation."""
        self.active_append = max(0, self.active_append - 1)

    def get_contention_factor(self, op_type: str) -> float:
        """Get latency multiplier based on current contention level.

        Uses linear interpolation from 1.0 at 1 concurrent operation to
        the provider's contention_scaling value at 16 concurrent operations.

        Args:
            op_type: "cas" or "append"

        Returns:
            Multiplier for base latency (1.0 if no scaling configured)
        """
        if not CONTENTION_SCALING_ENABLED:
            return 1.0

        if op_type == "cas":
            n = max(1, self.active_cas)
            scaling = CATALOG_CONTENTION_SCALING.get('cas', 1.0)
        else:
            n = max(1, self.active_append)
            scaling = CATALOG_CONTENTION_SCALING.get('append', 1.0)

        if scaling is None or scaling == 1.0:
            return 1.0

        # Linear interpolation: factor = 1 + (scaling - 1) * (n - 1) / 15
        # At n=1: factor=1.0, at n=16: factor=scaling
        return 1.0 + (scaling - 1.0) * min(n - 1, 15) / 15.0

    def reset(self):
        """Reset counters (for testing)."""
        self.active_cas = 0
        self.active_append = 0


# Global contention tracker instance
CONTENTION_TRACKER = ContentionTracker()


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
    # Success: mu=11.04, sigma=0.14, median=60.8ms
    # Failure: median=65ms (1.07x), similar distribution shape
    "s3": {
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
    # CAS success: median=22.4ms, sigma=0.22
    # CAS failure: median=21.1ms (0.95x), similar shape
    # Append success: median=20.5ms, sigma=0.25
    # Append failure: median=22.6ms (1.09x), similar shape
    "s3x": {
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
    # CAS success: median=93ms, sigma=0.82 (very heavy tails)
    # CAS failure: median=75ms (0.81x), failures actually faster
    # Append success: median=87ms, sigma=0.28
    # Append failure: median=2072ms, sigma=0.68 (COMPLETELY DIFFERENT DISTRIBUTION!)
    "azure": {
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
    # CAS success: median=64ms, sigma=0.73
    # CAS failure: median=82ms (1.28x)
    # Append success: median=70ms, sigma=0.23
    # Append failure: median=2534ms, sigma=0.65 (COMPLETELY DIFFERENT DISTRIBUTION!)
    "azurex": {
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
    # CAS success: median=170ms, sigma=0.91 (extremely heavy tails)
    # CAS failure: mean=7111ms, estimated median≈4000-5000ms based on heavy tails
    "gcp": {
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

    return result


def partition_tables_into_groups(n_tables: int, n_groups: int, distribution: str, longtail_params: dict) -> tuple[dict, dict]:
    """Partition tables into groups.

    Returns:
        table_to_group: dict mapping table ID to group ID
        group_to_tables: dict mapping group ID to list of table IDs
    """
    if n_groups > n_tables:
        logger.warning(f"Number of groups ({n_groups}) exceeds number of tables ({n_tables}). Setting n_groups = n_tables.")
        n_groups = n_tables

    table_to_group = {}
    group_to_tables = {g: [] for g in range(n_groups)}

    if distribution == "uniform":
        # Uniform distribution: distribute tables evenly across groups
        tables_per_group = n_tables // n_groups
        remainder = n_tables % n_groups

        table_id = 0
        for group_id in range(n_groups):
            # Give extra table to first 'remainder' groups
            group_size = tables_per_group + (1 if group_id < remainder else 0)
            for _ in range(group_size):
                table_to_group[table_id] = group_id
                group_to_tables[group_id].append(table_id)
                table_id += 1

    elif distribution == "longtail":
        # Longtail distribution: one large group, few medium groups, many small groups
        large_frac = longtail_params.get("large_group_fraction", 0.5)
        medium_count = longtail_params.get("medium_groups_count", 3)
        medium_frac = longtail_params.get("medium_group_fraction", 0.3)

        # Calculate sizes
        large_size = int(n_tables * large_frac)
        remaining_after_large = n_tables - large_size
        medium_total_size = int(remaining_after_large * medium_frac)
        small_total_size = remaining_after_large - medium_total_size

        # Ensure we have enough groups
        if n_groups < 1 + medium_count:
            logger.warning(f"Not enough groups ({n_groups}) for longtail distribution (need at least {1 + medium_count}). Using uniform distribution.")
            return partition_tables_into_groups(n_tables, n_groups, "uniform", {})

        medium_size = medium_total_size // medium_count if medium_count > 0 else 0
        small_groups_count = n_groups - 1 - medium_count
        small_size = small_total_size // small_groups_count if small_groups_count > 0 else 0

        table_id = 0
        group_id = 0

        # Large group
        for _ in range(large_size):
            table_to_group[table_id] = group_id
            group_to_tables[group_id].append(table_id)
            table_id += 1
        group_id += 1

        # Medium groups
        for _ in range(medium_count):
            for _ in range(medium_size):
                if table_id < n_tables:
                    table_to_group[table_id] = group_id
                    group_to_tables[group_id].append(table_id)
                    table_id += 1
            group_id += 1

        # Small groups (distribute remaining tables)
        while table_id < n_tables:
            for gid in range(group_id, n_groups):
                if table_id < n_tables:
                    table_to_group[table_id] = gid
                    group_to_tables[gid].append(table_id)
                    table_id += 1

    else:
        raise ValueError(f"Unknown group size distribution: {distribution}")

    return table_to_group, group_to_tables


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
    import hashlib
    import json
    from pathlib import Path

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


def configure_from_toml(config_file: str):
    global N_TABLES, N_GROUPS, GROUP_SIZE_DIST, LONGTAIL_PARAMS
    global TABLE_TO_GROUP, GROUP_TO_TABLES
    global T_CAS, T_METADATA_ROOT, T_MANIFEST_LIST, T_MANIFEST_FILE
    global T_MIN_RUNTIME, T_RUNTIME_MU, T_RUNTIME_SIGMA
    global N_TBL_PMF, TBL_R_PMF, N_TBL_W_PMF, N_TXN_RETRY
    global SIM_DURATION_MS, SIM_OUTPUT_PATH, SIM_SEED
    global INTER_ARRIVAL_DIST, INTER_ARRIVAL_PARAMS
    global MAX_PARALLEL, MIN_LATENCY
    global REAL_CONFLICT_PROBABILITY, CONFLICTING_MANIFESTS_DIST, CONFLICTING_MANIFESTS_PARAMS
    global RETRY_BACKOFF_ENABLED, RETRY_BACKOFF_BASE_MS, RETRY_BACKOFF_MULTIPLIER, RETRY_BACKOFF_MAX_MS, RETRY_BACKOFF_JITTER
    global EXPERIMENT_LABEL
    global CATALOG_MODE, COMPACTION_THRESHOLD, COMPACTION_MAX_ENTRIES, LOG_ENTRY_SIZE
    global TABLE_METADATA_INLINED, T_TABLE_METADATA_R, T_TABLE_METADATA_W
    global MANIFEST_LIST_MODE, MANIFEST_LIST_SEAL_THRESHOLD, MANIFEST_LIST_ENTRY_SIZE
    global T_APPEND, T_LOG_ENTRY_READ, T_COMPACTION
    global LATENCY_DISTRIBUTION
    global STORAGE_PROVIDER, CATALOG_BACKEND
    global CONTENTION_SCALING_ENABLED, CATALOG_CONTENTION_SCALING

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    # Load simulation parameters
    SIM_DURATION_MS = config["simulation"]["duration_ms"]
    SIM_OUTPUT_PATH = config["simulation"]["output_path"]
    SIM_SEED = config["simulation"].get("seed")

    # Load experiment parameters
    EXPERIMENT_LABEL = config.get("experiment", {}).get("label")

    # Load basic integer configuration
    N_TABLES = config["catalog"]["num_tables"]
    N_GROUPS = config["catalog"].get("num_groups", 1)
    GROUP_SIZE_DIST = config["catalog"].get("group_size_distribution", "uniform")

    # Load longtail parameters
    LONGTAIL_PARAMS = {}
    if "longtail" in config["catalog"]:
        LONGTAIL_PARAMS = {
            "large_group_fraction": config["catalog"]["longtail"].get("large_group_fraction", 0.5),
            "medium_groups_count": config["catalog"]["longtail"].get("medium_groups_count", 3),
            "medium_group_fraction": config["catalog"]["longtail"].get("medium_group_fraction", 0.3)
        }

    # Load append mode parameters
    CATALOG_MODE = config["catalog"].get("mode", "cas")
    COMPACTION_THRESHOLD = config["catalog"].get("compaction_threshold", 16 * 1024 * 1024)  # 16MB default
    COMPACTION_MAX_ENTRIES = config["catalog"].get("compaction_max_entries", 0)  # 0 = disabled (size only)
    LOG_ENTRY_SIZE = config["catalog"].get("log_entry_size", 100)  # 100 bytes default

    # Load storage and catalog configuration
    MAX_PARALLEL = config["storage"]["max_parallel"]
    MIN_LATENCY = config["storage"]["min_latency"]

    storage_cfg = config.get("storage", {})
    catalog_cfg = config.get("catalog", {})

    # Provider and backend configuration
    STORAGE_PROVIDER = storage_cfg.get("provider", None)
    CATALOG_BACKEND = catalog_cfg.get("backend", "storage")  # Default: catalog ops in storage

    # Global latency distribution setting (can be overridden per-operation)
    LATENCY_DISTRIBUTION = storage_cfg.get("latency_distribution", "normal")

    # Hardcoded defaults (normal distribution for backward compatibility)
    default_cas = {'mean': 100.0, 'stddev': 10.0, 'distribution': 'normal'}
    default_metadata_root = {'mean': 1.0, 'stddev': 0.1, 'distribution': 'normal'}
    default_manifest = {'mean': 50.0, 'stddev': 5.0, 'distribution': 'normal'}
    default_manifest_write = {'mean': 60.0, 'stddev': 6.0, 'distribution': 'normal'}
    default_append = {'mean': 50.0, 'stddev': 5.0, 'distribution': 'normal'}
    default_log_read = {'mean': 5.0, 'stddev': 1.0, 'distribution': 'normal'}
    default_compaction = {'mean': 200.0, 'stddev': 20.0, 'distribution': 'normal'}

    # Apply provider profile defaults if specified
    if STORAGE_PROVIDER and STORAGE_PROVIDER in PROVIDER_PROFILES:
        provider_defaults = apply_provider_defaults(
            STORAGE_PROVIDER, storage_cfg, CATALOG_BACKEND, catalog_cfg
        )
    else:
        provider_defaults = {}

    # === STORAGE OPERATIONS (always from storage, never from catalog service) ===

    # T_MANIFEST_LIST: use provider defaults if available, else explicit config, else hardcoded
    if 'T_MANIFEST_LIST' in provider_defaults and "T_MANIFEST_LIST" not in storage_cfg:
        T_MANIFEST_LIST = provider_defaults['T_MANIFEST_LIST']
    elif "T_MANIFEST_LIST" in storage_cfg:
        T_MANIFEST_LIST = {
            'read': parse_latency_config(storage_cfg["T_MANIFEST_LIST"].get("read", {}), default_manifest),
            'write': parse_latency_config(storage_cfg["T_MANIFEST_LIST"].get("write", {}), default_manifest_write)
        }
    else:
        T_MANIFEST_LIST = {'read': default_manifest.copy(), 'write': default_manifest_write.copy()}

    # T_MANIFEST_FILE: same logic
    if 'T_MANIFEST_FILE' in provider_defaults and "T_MANIFEST_FILE" not in storage_cfg:
        T_MANIFEST_FILE = provider_defaults['T_MANIFEST_FILE']
    elif "T_MANIFEST_FILE" in storage_cfg:
        T_MANIFEST_FILE = {
            'read': parse_latency_config(storage_cfg["T_MANIFEST_FILE"].get("read", {}), default_manifest),
            'write': parse_latency_config(storage_cfg["T_MANIFEST_FILE"].get("write", {}), default_manifest_write)
        }
    else:
        T_MANIFEST_FILE = {'read': default_manifest.copy(), 'write': default_manifest_write.copy()}

    # T_METADATA_ROOT: typically fast (catalog pointer), explicit config or default
    if "T_METADATA_ROOT" in storage_cfg:
        T_METADATA_ROOT = {
            'read': parse_latency_config(storage_cfg["T_METADATA_ROOT"].get("read", {}), default_metadata_root),
            'write': parse_latency_config(storage_cfg["T_METADATA_ROOT"].get("write", {}), default_metadata_root)
        }
    else:
        T_METADATA_ROOT = {'read': default_metadata_root.copy(), 'write': default_metadata_root.copy()}

    # === CATALOG OPERATIONS (depend on backend) ===

    # Determine catalog latency source based on backend
    if CATALOG_BACKEND == "service":
        # Catalog service has its own config section
        service_cfg = catalog_cfg.get("service", {})
        service_provider = service_cfg.get("provider", STORAGE_PROVIDER)

        # T_CAS: use service config, or service provider, or storage provider
        if "T_CAS" in service_cfg:
            T_CAS = parse_latency_config(service_cfg["T_CAS"], default_cas)
        elif service_provider and service_provider in PROVIDER_PROFILES:
            T_CAS = get_provider_latency_config(service_provider, 'cas') or default_cas
        elif "T_CAS" in storage_cfg:
            T_CAS = parse_latency_config(storage_cfg["T_CAS"], default_cas)
        else:
            T_CAS = default_cas.copy()

        # T_APPEND: same logic
        if "T_APPEND" in service_cfg:
            T_APPEND = parse_latency_config(service_cfg["T_APPEND"], default_append)
        elif service_provider and service_provider in PROVIDER_PROFILES:
            T_APPEND = get_provider_latency_config(service_provider, 'append') or default_append
        elif "T_APPEND" in storage_cfg:
            T_APPEND = parse_latency_config(storage_cfg["T_APPEND"], default_append)
        else:
            T_APPEND = default_append.copy()

    elif CATALOG_BACKEND == "fifo_queue":
        # FIFO queue: append from queue config, CAS for compaction from storage
        queue_cfg = catalog_cfg.get("fifo_queue", {})

        # T_APPEND: queue append latency
        if "append" in queue_cfg:
            T_APPEND = parse_latency_config(queue_cfg["append"], default_append)
        elif 'T_APPEND' in provider_defaults:
            T_APPEND = provider_defaults['T_APPEND']
        else:
            T_APPEND = default_append.copy()

        # T_CAS: compaction uses storage
        if "T_CAS" in storage_cfg:
            T_CAS = parse_latency_config(storage_cfg["T_CAS"], default_cas)
        elif 'T_CAS' in provider_defaults:
            T_CAS = provider_defaults['T_CAS']
        else:
            T_CAS = default_cas.copy()

    else:  # backend == "storage" (default)
        # Catalog in storage: use provider defaults or explicit config
        if 'T_CAS' in provider_defaults and "T_CAS" not in storage_cfg:
            T_CAS = provider_defaults['T_CAS']
        elif "T_CAS" in storage_cfg:
            T_CAS = parse_latency_config(storage_cfg["T_CAS"], default_cas)
        else:
            T_CAS = default_cas.copy()

        if 'T_APPEND' in provider_defaults and "T_APPEND" not in storage_cfg:
            T_APPEND = provider_defaults['T_APPEND']
        elif "T_APPEND" in storage_cfg:
            T_APPEND = parse_latency_config(storage_cfg["T_APPEND"], default_append)
        else:
            T_APPEND = default_append.copy()

    # === OTHER STORAGE LATENCIES ===

    T_LOG_ENTRY_READ = parse_latency_config(
        storage_cfg.get("T_LOG_ENTRY_READ", {}), default_log_read
    )
    T_COMPACTION = parse_latency_config(
        storage_cfg.get("T_COMPACTION", {}), default_compaction
    )

    # Table metadata inlining configuration
    TABLE_METADATA_INLINED = config.get("catalog", {}).get("table_metadata_inlined", True)
    table_metadata_cfg = storage_cfg.get("T_TABLE_METADATA", {})
    T_TABLE_METADATA_R = parse_latency_config(
        table_metadata_cfg.get("read", {}),
        {'mean': 20.0, 'stddev': 5.0, 'distribution': 'normal'}
    )
    T_TABLE_METADATA_W = parse_latency_config(
        table_metadata_cfg.get("write", {}),
        {'mean': 30.0, 'stddev': 5.0, 'distribution': 'normal'}
    )

    # Manifest list append configuration
    MANIFEST_LIST_SEAL_THRESHOLD = config.get("transaction", {}).get("manifest_list_seal_threshold", 0)  # 0 = disabled
    MANIFEST_LIST_ENTRY_SIZE = config.get("transaction", {}).get("manifest_list_entry_size", 50)  # 50 bytes default

    # Load runtime-related configuration
    N_TXN_RETRY = config["transaction"]["retry"]
    T_MIN_RUNTIME = config["transaction"]["runtime"]["min"]
    mean = config["transaction"]["runtime"]["mean"]
    sigma = config["transaction"]["runtime"]["sigma"]
    T_RUNTIME_MU, T_RUNTIME_SIGMA = lognormal_params_from_mean_and_sigma(mean, sigma)

    # Load manifest list mode
    MANIFEST_LIST_MODE = config.get("transaction", {}).get("manifest_list_mode", "rewrite")

    # Load inter-arrival distribution parameters
    INTER_ARRIVAL_DIST = config["transaction"]["inter_arrival"]["distribution"]
    INTER_ARRIVAL_PARAMS = {}
    if INTER_ARRIVAL_DIST == "exponential":
        INTER_ARRIVAL_PARAMS["scale"] = config["transaction"]["inter_arrival"]["scale"]
    elif INTER_ARRIVAL_DIST == "uniform":
        INTER_ARRIVAL_PARAMS["min"] = config["transaction"]["inter_arrival"]["min"]
        INTER_ARRIVAL_PARAMS["max"] = config["transaction"]["inter_arrival"]["max"]
    elif INTER_ARRIVAL_DIST == "normal":
        INTER_ARRIVAL_PARAMS["mean"] = config["transaction"]["inter_arrival"]["mean"]
        INTER_ARRIVAL_PARAMS["std_dev"] = config["transaction"]["inter_arrival"]["std_dev"]
    elif INTER_ARRIVAL_DIST == "fixed":
        INTER_ARRIVAL_PARAMS["value"] = config["transaction"]["inter_arrival"]["value"]

    # Load parameters for PMFs
    ntbl_exponent = config.get("transaction", {}).get("ntable", {}).get("zipf", 2.0)
    tblr_exponent = config.get("transaction", {}).get("seltbl", {}).get("zipf", 1.4)
    ntblw_exponent = config.get("transaction", {}).get("seltblw", {}).get("zipf", 1.2)

    # Generate PMFs
    N_TBL_PMF = truncated_zipf_pmf(N_TABLES, ntbl_exponent)
    TBL_R_PMF = truncated_zipf_pmf(N_TABLES, tblr_exponent)
    N_TBL_W_PMF = [truncated_zipf_pmf(k, ntblw_exponent) for k in range(0, N_TABLES + 1)]

    # Load conflict resolution parameters
    REAL_CONFLICT_PROBABILITY = config.get("transaction", {}).get("real_conflict_probability", 0.0)
    CONFLICTING_MANIFESTS_DIST = config.get("transaction", {}).get("conflicting_manifests", {}).get("distribution", "exponential")
    CONFLICTING_MANIFESTS_PARAMS = {
        "mean": config.get("transaction", {}).get("conflicting_manifests", {}).get("mean", 3.0),
        "min": config.get("transaction", {}).get("conflicting_manifests", {}).get("min", 1),
        "max": config.get("transaction", {}).get("conflicting_manifests", {}).get("max", 10),
        "value": config.get("transaction", {}).get("conflicting_manifests", {}).get("value", 3),
    }

    # Load retry backoff parameters
    RETRY_BACKOFF_ENABLED = config.get("transaction", {}).get("retry_backoff", {}).get("enabled", False)
    RETRY_BACKOFF_BASE_MS = config.get("transaction", {}).get("retry_backoff", {}).get("base_ms", 10.0)
    RETRY_BACKOFF_MULTIPLIER = config.get("transaction", {}).get("retry_backoff", {}).get("multiplier", 2.0)
    RETRY_BACKOFF_MAX_MS = config.get("transaction", {}).get("retry_backoff", {}).get("max_ms", 5000.0)
    RETRY_BACKOFF_JITTER = config.get("transaction", {}).get("retry_backoff", {}).get("jitter", 0.1)

    # === CONTENTION SCALING ===

    # Enable contention scaling if a provider is used and contention_scaling_enabled is not False
    contention_enabled = storage_cfg.get("contention_scaling_enabled", None)
    if contention_enabled is not None:
        CONTENTION_SCALING_ENABLED = contention_enabled
    elif STORAGE_PROVIDER and STORAGE_PROVIDER in PROVIDER_PROFILES:
        # Auto-enable when using a provider profile
        CONTENTION_SCALING_ENABLED = True
    else:
        CONTENTION_SCALING_ENABLED = False

    # Get contention scaling factors from provider
    if CONTENTION_SCALING_ENABLED:
        # Determine which provider to use for contention scaling
        if CATALOG_BACKEND == "service":
            service_cfg = catalog_cfg.get("service", {})
            scaling_provider = service_cfg.get("provider", STORAGE_PROVIDER)
        else:
            scaling_provider = STORAGE_PROVIDER

        if scaling_provider and scaling_provider in PROVIDER_PROFILES:
            profile = PROVIDER_PROFILES[scaling_provider]
            CATALOG_CONTENTION_SCALING = profile.get('contention_scaling', {}) or {}
        else:
            CATALOG_CONTENTION_SCALING = {}
    else:
        CATALOG_CONTENTION_SCALING = {}

    # Reset contention tracker
    CONTENTION_TRACKER.reset()

    # NOTE: Table partitioning is done in CLI after seed is set for determinism


def validate_config(config: dict) -> list[str]:
    """Validate configuration and return list of errors.

    Returns:
        List of validation error messages. Empty list if configuration is valid.
    """
    errors = []

    # Validate catalog configuration
    num_tables = config.get('catalog', {}).get('num_tables', 0)
    num_groups = config.get('catalog', {}).get('num_groups', 1)

    if num_tables <= 0:
        errors.append("catalog.num_tables must be > 0")

    if num_groups <= 0:
        errors.append("catalog.num_groups must be > 0")

    if num_groups > num_tables:
        errors.append(f"catalog.num_groups ({num_groups}) cannot exceed num_tables ({num_tables})")

    # Validate group size distribution
    group_dist = config.get('catalog', {}).get('group_size_distribution', 'uniform')
    if group_dist not in ['uniform', 'longtail']:
        errors.append(f"catalog.group_size_distribution must be 'uniform' or 'longtail', got '{group_dist}'")

    # Validate longtail parameters if using longtail distribution
    if group_dist == 'longtail' and 'longtail' in config.get('catalog', {}):
        large_frac = config['catalog']['longtail'].get('large_group_fraction', 0.5)
        medium_frac = config['catalog']['longtail'].get('medium_group_fraction', 0.3)
        medium_count = config['catalog']['longtail'].get('medium_groups_count', 3)

        if not (0 < large_frac < 1):
            errors.append(f"longtail.large_group_fraction must be in (0, 1), got {large_frac}")

        if not (0 <= medium_frac < 1):
            errors.append(f"longtail.medium_group_fraction must be in [0, 1), got {medium_frac}")

        if large_frac + medium_frac >= 1.0:
            errors.append(f"longtail.large_group_fraction ({large_frac}) + medium_group_fraction ({medium_frac}) must be < 1.0")

        if medium_count < 0:
            errors.append(f"longtail.medium_groups_count must be >= 0, got {medium_count}")

        if num_groups > 1 and num_groups < 1 + medium_count:
            errors.append(f"longtail distribution requires at least {1 + medium_count} groups (1 large + {medium_count} medium), but num_groups = {num_groups}")

    # Validate storage configuration
    storage = config.get('storage', {})
    min_latency = storage.get('min_latency', 5)
    max_parallel = storage.get('max_parallel', 4)

    if min_latency < 0:
        errors.append(f"storage.min_latency must be >= 0, got {min_latency}")

    if max_parallel <= 0:
        errors.append(f"storage.max_parallel must be > 0, got {max_parallel}")

    # Validate CAS latency
    if 'T_CAS' in storage:
        cas_mean = storage['T_CAS'].get('mean', 100)
        cas_stddev = storage['T_CAS'].get('stddev', 10)

        if cas_mean <= 0:
            errors.append(f"T_CAS.mean must be > 0, got {cas_mean}")

        if cas_stddev < 0:
            errors.append(f"T_CAS.stddev must be >= 0, got {cas_stddev}")

        if min_latency > cas_mean:
            errors.append(f"storage.min_latency ({min_latency}) should not exceed T_CAS.mean ({cas_mean})")

    # Validate transaction configuration
    txn = config.get('transaction', {})
    retry = txn.get('retry', 10)

    if retry < 0:
        errors.append(f"transaction.retry must be >= 0, got {retry}")

    # Validate inter-arrival distribution
    if 'inter_arrival' in txn:
        dist = txn['inter_arrival'].get('distribution', 'exponential')
        if dist not in ['fixed', 'exponential', 'uniform', 'normal']:
            errors.append(f"inter_arrival.distribution must be one of [fixed, exponential, uniform, normal], got '{dist}'")

        if dist == 'exponential' and 'scale' in txn['inter_arrival']:
            scale = txn['inter_arrival']['scale']
            if scale <= 0:
                errors.append(f"inter_arrival.scale must be > 0, got {scale}")

        if dist == 'fixed' and 'value' in txn['inter_arrival']:
            value = txn['inter_arrival']['value']
            if value <= 0:
                errors.append(f"inter_arrival.value must be > 0, got {value}")

        if dist == 'uniform':
            ia_min = txn['inter_arrival'].get('min', 0)
            ia_max = txn['inter_arrival'].get('max', 1000)
            if ia_min >= ia_max:
                errors.append(f"inter_arrival.min ({ia_min}) must be < inter_arrival.max ({ia_max})")

    # Validate conflict resolution configuration
    real_conflict_prob = txn.get('real_conflict_probability', 0.0)
    if real_conflict_prob < 0 or real_conflict_prob > 1:
        errors.append(f"transaction.real_conflict_probability must be in [0, 1], got {real_conflict_prob}")

    if 'conflicting_manifests' in txn:
        cm = txn['conflicting_manifests']
        dist = cm.get('distribution', 'exponential')

        if dist not in ['fixed', 'exponential', 'uniform']:
            errors.append(f"conflicting_manifests.distribution must be one of [fixed, exponential, uniform], got '{dist}'")

        cm_min = cm.get('min', 1)
        cm_max = cm.get('max', 10)

        if cm_min <= 0:
            errors.append(f"conflicting_manifests.min must be > 0, got {cm_min}")

        if cm_max < cm_min:
            errors.append(f"conflicting_manifests.max ({cm_max}) must be >= min ({cm_min})")

        if dist == 'exponential':
            mean = cm.get('mean', 3.0)
            if mean <= 0:
                errors.append(f"conflicting_manifests.mean must be > 0, got {mean}")

        if dist == 'fixed':
            value = cm.get('value', 3)
            if value <= 0:
                errors.append(f"conflicting_manifests.value must be > 0, got {value}")

    # Validate simulation configuration
    sim = config.get('simulation', {})
    duration = sim.get('duration_ms', 0)

    if duration <= 0:
        errors.append(f"simulation.duration_ms must be > 0, got {duration}")

    return errors


def generate_inter_arrival_time():
    """Generate inter-arrival time based on configured distribution."""
    if INTER_ARRIVAL_DIST == "fixed":
        return INTER_ARRIVAL_PARAMS["value"]
    elif INTER_ARRIVAL_DIST == "exponential":
        return np.random.exponential(scale=INTER_ARRIVAL_PARAMS["scale"])
    elif INTER_ARRIVAL_DIST == "uniform":
        return np.random.uniform(
            low=INTER_ARRIVAL_PARAMS["min"],
            high=INTER_ARRIVAL_PARAMS["max"]
        )
    elif INTER_ARRIVAL_DIST == "normal":
        # Ensure non-negative values for normal distribution
        return max(0, np.random.normal(
            loc=INTER_ARRIVAL_PARAMS["mean"],
            scale=INTER_ARRIVAL_PARAMS["std_dev"]
        ))
    else:
        raise ValueError(f"Unknown inter-arrival distribution: {INTER_ARRIVAL_DIST}")


def generate_latency(mean: float, stddev: float) -> float:
    """Generate storage operation latency from normal distribution (legacy).

    Enforces minimum latency to prevent unrealistic zero or near-zero values.
    """
    return max(MIN_LATENCY, np.random.normal(loc=mean, scale=stddev))


def generate_latency_lognormal(mu: float, sigma: float) -> float:
    """Generate storage operation latency from lognormal distribution.

    Args:
        mu: Mean of underlying normal distribution (log-scale).
            For lognormal, median = exp(mu).
        sigma: Std dev of underlying normal distribution (log-scale).
            Higher sigma = heavier tail.

    Returns:
        Latency in milliseconds.
    """
    return max(MIN_LATENCY, np.random.lognormal(mean=mu, sigma=sigma))


def generate_latency_from_config(params: dict) -> float:
    """Generate latency based on configuration (supports both distributions).

    Args:
        params: Dict with either:
            - {'mu': float, 'sigma': float, 'distribution': 'lognormal'}
            - {'mean': float, 'stddev': float, 'distribution': 'normal'}

    Returns:
        Latency in milliseconds.
    """
    dist = params.get('distribution', 'normal')
    if dist == 'lognormal':
        return generate_latency_lognormal(params['mu'], params['sigma'])
    else:
        return generate_latency(params['mean'], params['stddev'])


def get_cas_latency(success: bool = True) -> float:
    """Get CAS operation latency.

    Args:
        success: If True, draw from success distribution; else use failure distribution.
                 Based on measurements, failed operations can have completely different
                 distributions (e.g., GCP CAS failures are 13x slower with heavier tails).

    Returns:
        Latency in milliseconds.
    """
    if not success and 'failure' in T_CAS:
        # Use separate failure distribution
        failure_config = {
            'mu': lognormal_mu_from_median(T_CAS['failure']['median']),
            'sigma': T_CAS['failure']['sigma'],
            'distribution': 'lognormal'
        }
        base = generate_latency_from_config(failure_config)
    else:
        base = generate_latency_from_config(T_CAS)
        # Legacy: apply failure multiplier if no separate distribution
        if not success:
            multiplier = T_CAS.get('failure_multiplier', 1.0)
            base *= multiplier

    # Apply contention scaling
    if CONTENTION_SCALING_ENABLED:
        base *= CONTENTION_TRACKER.get_contention_factor("cas")

    return base


def get_metadata_root_latency(operation: str) -> float:
    """Get metadata root latency for read or write operation."""
    params = T_METADATA_ROOT[operation]
    return generate_latency_from_config(params)


def get_manifest_list_latency(operation: str) -> float:
    """Get manifest list latency for read or write operation."""
    params = T_MANIFEST_LIST[operation]
    return generate_latency_from_config(params)


def get_manifest_file_latency(operation: str) -> float:
    """Get manifest file latency for read or write operation."""
    params = T_MANIFEST_FILE[operation]
    return generate_latency_from_config(params)


def get_append_latency(success: bool = True) -> float:
    """Get append operation latency (for catalog log append).

    Args:
        success: If True, draw from success distribution; else use failure distribution.
                 CRITICAL: Azure append failures have completely different distribution!
                 Success median=87ms, Failure median=2072ms (24x), with different shape.

    Returns:
        Latency in milliseconds.
    """
    if not success and 'failure' in T_APPEND:
        # Use separate failure distribution - critical for Azure!
        failure_config = {
            'mu': lognormal_mu_from_median(T_APPEND['failure']['median']),
            'sigma': T_APPEND['failure']['sigma'],
            'distribution': 'lognormal'
        }
        base = generate_latency_from_config(failure_config)
    else:
        base = generate_latency_from_config(T_APPEND)
        # Legacy: apply failure multiplier if no separate distribution
        if not success:
            multiplier = T_APPEND.get('failure_multiplier', 1.0)
            base *= multiplier

    # Apply contention scaling
    if CONTENTION_SCALING_ENABLED:
        base *= CONTENTION_TRACKER.get_contention_factor("append")

    return base


def get_log_entry_read_latency() -> float:
    """Get per-entry log read latency."""
    return generate_latency_from_config(T_LOG_ENTRY_READ)


def get_compaction_latency() -> float:
    """Get compaction CAS latency (larger payload than normal CAS)."""
    return generate_latency_from_config(T_COMPACTION)


def get_table_metadata_latency(operation: str) -> float:
    """Get table metadata latency for read or write operation."""
    if operation == 'read':
        return generate_latency_from_config(T_TABLE_METADATA_R)
    else:
        return generate_latency_from_config(T_TABLE_METADATA_W)


def sample_conflicting_manifests() -> int:
    """Sample number of conflicting manifest files from configured distribution.

    Returns:
        Number of manifest files that need to be read and merged during
        real conflict resolution.
    """
    dist = CONFLICTING_MANIFESTS_DIST
    params = CONFLICTING_MANIFESTS_PARAMS

    if dist == "fixed":
        return int(params['value'])
    elif dist == "exponential":
        value = np.random.exponential(params['mean'])
        return max(params['min'], min(params['max'], int(value)))
    elif dist == "uniform":
        return np.random.randint(params['min'], params['max'] + 1)
    else:
        raise ValueError(f"Unknown conflicting_manifests distribution: {dist}")


def calculate_backoff_time(retry_number: int) -> float:
    """Calculate exponential backoff time with jitter.

    Args:
        retry_number: Current retry attempt (1-indexed)

    Returns:
        Backoff time in milliseconds
    """
    if not RETRY_BACKOFF_ENABLED:
        return 0.0

    # Exponential backoff: base * multiplier^(retry_number - 1)
    backoff = RETRY_BACKOFF_BASE_MS * (RETRY_BACKOFF_MULTIPLIER ** (retry_number - 1))

    # Cap at maximum
    backoff = min(backoff, RETRY_BACKOFF_MAX_MS)

    # Add jitter: random factor between (1 - jitter) and (1 + jitter)
    if RETRY_BACKOFF_JITTER > 0:
        jitter_factor = 1.0 + np.random.uniform(-RETRY_BACKOFF_JITTER, RETRY_BACKOFF_JITTER)
        backoff *= jitter_factor

    return max(0, backoff)


def print_configuration():
    """Print configuration summary."""
    print("\n" + "="*70)
    print("  ENDIVE SIMULATOR CONFIGURATION")
    print("="*70)

    print("\n[Simulation]")
    print(f"  Duration:     {SIM_DURATION_MS:,} ms ({SIM_DURATION_MS/1000:.1f} seconds)")
    print(f"  Output:       {SIM_OUTPUT_PATH}")
    if SIM_SEED is not None:
        print(f"  Random Seed:  {SIM_SEED}")
    else:
        print(f"  Random Seed:  <auto-generated>")

    print("\n[Catalog]")
    print(f"  Tables:       {N_TABLES}")
    print(f"  Groups:       {N_GROUPS}")
    if N_GROUPS > 1:
        print(f"  Distribution: {GROUP_SIZE_DIST}")
        # GROUP_TO_TABLES will be populated after seed is set
        if N_GROUPS == N_TABLES:
            print(f"  Conflicts:    Table-level only (no multi-table transactions)")

    print("\n[Transaction]")
    print(f"  Max Retries:  {N_TXN_RETRY}")
    print(f"  Runtime:      {T_MIN_RUNTIME}ms min, {int(np.exp(T_RUNTIME_MU + T_RUNTIME_SIGMA**2/2))}ms mean (lognormal)")

    print("\n[Workload]")
    print(f"  Inter-arrival: {INTER_ARRIVAL_DIST}")
    if INTER_ARRIVAL_DIST == "exponential":
        print(f"    Scale:      {INTER_ARRIVAL_PARAMS['scale']:.1f}ms")
        print(f"    (Mean rate: ~{1000/INTER_ARRIVAL_PARAMS['scale']:.2f} txn/sec)")
    elif INTER_ARRIVAL_DIST == "fixed":
        print(f"    Value:      {INTER_ARRIVAL_PARAMS['value']:.1f}ms")
        print(f"    (Rate:      {1000/INTER_ARRIVAL_PARAMS['value']:.2f} txn/sec)")

    print("\n[Storage Latencies (ms, mean±stddev)]")
    print(f"  Min Latency:  {MIN_LATENCY:.1f}ms")
    print(f"  CAS:          {T_CAS['mean']:.1f}±{T_CAS['stddev']:.1f}")
    print(f"  Metadata R/W: {T_METADATA_ROOT['read']['mean']:.1f}±{T_METADATA_ROOT['read']['stddev']:.1f} / "
          f"{T_METADATA_ROOT['write']['mean']:.1f}±{T_METADATA_ROOT['write']['stddev']:.1f}")
    print(f"  Manifest L:   {T_MANIFEST_LIST['read']['mean']:.1f}±{T_MANIFEST_LIST['read']['stddev']:.1f} / "
          f"{T_MANIFEST_LIST['write']['mean']:.1f}±{T_MANIFEST_LIST['write']['stddev']:.1f}")
    print(f"  Manifest F:   {T_MANIFEST_FILE['read']['mean']:.1f}±{T_MANIFEST_FILE['read']['stddev']:.1f} / "
          f"{T_MANIFEST_FILE['write']['mean']:.1f}±{T_MANIFEST_FILE['write']['stddev']:.1f}")
    print(f"  Max Parallel: {MAX_PARALLEL}")

    print("\n" + "="*70 + "\n")


def confirm_run() -> bool:
    """Ask user to confirm simulation run."""
    response = input("Proceed with simulation? [Y/n]: ").strip().lower()
    return response in ['', 'y', 'yes']


class ConflictResolver:
    """Handles conflict resolution for failed CAS operations.

    When a transaction's CAS fails, this class orchestrates:
    1. Calculating how many snapshots behind the transaction is
    2. Reading the appropriate number of manifest lists
    3. Merging conflicts for affected tables
    4. Updating the transaction's write set for retry
    """

    @staticmethod
    def calculate_snapshots_behind(txn: 'Txn', catalog: 'Catalog') -> int:
        """Calculate how many snapshots the transaction is behind.

        Returns: n where current catalog is at S_{i+n} and transaction is at S_i
        """
        return catalog.seq - txn.v_catalog_seq

    @staticmethod
    def read_manifest_lists(sim, n_snapshots: int, txn_id: int):
        """Read n manifest lists in batches respecting MAX_PARALLEL limit.

        Yields timeout events for simulating parallel I/O with batching.
        """
        if n_snapshots <= 0:
            return

        logger.debug(f"{sim.now} TXN {txn_id} Reading {n_snapshots} manifest lists (max_parallel={MAX_PARALLEL})")

        # Process in batches of MAX_PARALLEL
        for batch_start in range(0, n_snapshots, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_snapshots - batch_start)
            # All reads in this batch happen in parallel, take max time
            batch_latencies = [get_manifest_list_latency('read') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))
            logger.debug(f"{sim.now} TXN {txn_id} Read batch of {batch_size} manifest lists")

    @staticmethod
    def merge_table_conflicts(sim, txn: 'Txn', v_catalog: dict, catalog=None):
        """Merge conflicts for tables that have changed.

        For each dirty table that has a different version in the catalog,
        determines if this is a false conflict (version changed but no data overlap)
        or a real conflict (overlapping data changes), and resolves accordingly.

        In ML+ mode:
        - False conflict: ML entry still valid (different partition), no ML update needed
        - Real conflict: ML entry needs update, re-append with merged data

        Args:
            sim: SimPy environment
            txn: Transaction object
            v_catalog: Current catalog state
            catalog: Catalog reference (needed for ML+ mode)

        Yields timeout events for each I/O operation.
        """
        for t, v in txn.v_dirty.items():
            if v_catalog[t] != v:
                # Determine if this is a real conflict
                is_real_conflict = np.random.random() < REAL_CONFLICT_PROBABILITY

                if is_real_conflict:
                    yield from ConflictResolver.resolve_real_conflict(sim, txn, t, v_catalog, catalog)
                    STATS.real_conflicts += 1
                else:
                    # False conflict: In ML+ mode, the tentative ML entry is still valid
                    # (different partition, no data overlap). No ML update needed.
                    yield from ConflictResolver.resolve_false_conflict(sim, txn, t, v_catalog)
                    STATS.false_conflicts += 1

    @staticmethod
    def resolve_false_conflict(sim, txn, table_id: int, v_catalog: dict):
        """Resolve false conflict (version changed, no data overlap).

        Manifest lists were already read in read_manifest_lists().
        Only need to read metadata to understand the new snapshot state.
        No manifest file operations required.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            v_catalog: Current catalog state
        """
        logger.debug(f"{sim.now} TXN {txn.id} Resolving false conflict for table {table_id}")

        # Read metadata root to understand new snapshot
        yield sim.timeout(get_metadata_root_latency('read'))

        # Update validation version (no file operations needed)
        txn.v_dirty[table_id] = v_catalog[table_id]

    @staticmethod
    def resolve_real_conflict(sim, txn, table_id: int, v_catalog: dict, catalog=None):
        """Resolve real conflict (overlapping data changes).

        Must read and rewrite manifest files to merge conflicting changes.
        The number of conflicting manifests is sampled from configuration.

        In ML+ mode, the original tentative ML entry is filtered by readers
        (txn not committed). After merging, we append a NEW entry with merged data.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            v_catalog: Current catalog state
            catalog: Catalog reference (needed for ML+ mode)
        """
        # Determine number of conflicting manifest files
        n_conflicting = sample_conflicting_manifests()

        logger.debug(f"{sim.now} TXN {txn.id} Resolving real conflict for table {table_id} "
                    f"({n_conflicting} conflicting manifests)")

        # Read metadata root
        yield sim.timeout(get_metadata_root_latency('read'))

        # Read manifest list (to get pointers to manifest files)
        yield sim.timeout(get_manifest_list_latency('read'))

        # Read conflicting manifest files (respects MAX_PARALLEL)
        for batch_start in range(0, n_conflicting, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
            batch_latencies = [get_manifest_file_latency('read') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))
            logger.debug(f"{sim.now} TXN {txn.id} Read batch of {batch_size} manifest files")

        # Track manifest file operations
        STATS.manifest_files_read += n_conflicting

        # Write merged manifest files (respects MAX_PARALLEL)
        for batch_start in range(0, n_conflicting, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
            batch_latencies = [get_manifest_file_latency('write') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))
            logger.debug(f"{sim.now} TXN {txn.id} Wrote batch of {batch_size} merged manifest files")

        # Track manifest file operations
        STATS.manifest_files_written += n_conflicting

        # Update manifest list: append (ML+ mode) or rewrite (traditional mode)
        if MANIFEST_LIST_MODE == "append" and catalog is not None:
            # ML+ mode: Append new entry with merged data
            # The original tentative entry is filtered (txn not committed yet)
            expected_offset = txn.v_ml_offset.get(table_id, 0)
            yield sim.timeout(get_append_latency())
            physical_success = catalog.try_ML_APPEND(sim, table_id, expected_offset, txn.id)

            # Physical retry if offset moved
            while not physical_success:
                STATS.manifest_append_physical_failure += 1
                if catalog.ml_sealed[table_id]:
                    # Sealed - rewrite
                    STATS.manifest_append_sealed_rewrite += 1
                    yield sim.timeout(get_manifest_list_latency('write'))
                    catalog.rewrite_manifest_list(table_id)
                    txn.v_ml_offset[table_id] = catalog.ml_offset[table_id]
                    break
                txn.v_ml_offset[table_id] = catalog.ml_offset[table_id]
                yield sim.timeout(get_append_latency())
                physical_success = catalog.try_ML_APPEND(sim, table_id, txn.v_ml_offset[table_id], txn.id)

            STATS.manifest_append_physical_success += 1
            logger.debug(f"{sim.now} TXN {txn.id} ML_APPEND table {table_id} after real conflict")
        else:
            # Traditional mode: Write updated manifest list
            yield sim.timeout(get_manifest_list_latency('write'))

        # Update validation version
        txn.v_dirty[table_id] = v_catalog[table_id]

    @staticmethod
    def update_write_set(txn: 'Txn', v_catalog: dict):
        """Update transaction's write set to next available version per table.

        Sets each written table to current_version + 1, preparing the
        transaction to attempt installing the next snapshot.
        """
        for t in txn.v_tblw.keys():
            txn.v_tblw[t] = v_catalog[t] + 1

class Catalog:
    def __init__(self, sim):
        self.sim = sim
        self.seq = 0
        self.tbl = [0] * N_TABLES
        # Track last committed transaction per group (for table-level conflicts when N_GROUPS == N_TABLES)
        self.group_seq = [0] * N_GROUPS if N_GROUPS > 1 else None
        # Per-table manifest list state for append mode
        self.ml_offset = [0] * N_TABLES  # Manifest list byte offset per table
        self.ml_sealed = [False] * N_TABLES  # Whether manifest list is sealed (needs rewrite)

    def try_CAS(self, sim, txn):
        """Attempt compare-and-swap for transaction commit.

        When N_GROUPS == N_TABLES, only check conflicts at table level.
        Otherwise, check catalog-level conflicts.
        """
        logger.debug(f"{sim.now} TXN {txn.id} CAS {self.seq} = {txn.v_catalog_seq} {txn.v_tblw}")
        logger.debug(f"{sim.now} TXN {txn.id} Catalog {self.tbl}")

        # Table-level conflicts when each table is its own group
        if N_GROUPS == N_TABLES:
            # Check if any of the tables we read/wrote have changed
            conflict = False
            for t in txn.v_dirty.keys():
                if self.tbl[t] != txn.v_dirty[t]:
                    conflict = True
                    break

            if not conflict:
                # No conflicts - commit
                for off, val in txn.v_tblw.items():
                    self.tbl[off] = val
                self.seq += 1
                logger.debug(f"{sim.now} TXN {txn.id} CASOK (table-level)   {self.tbl}")
                return True
            return False
        else:
            # Catalog-level conflicts (original behavior)
            if self.seq == txn.v_catalog_seq:
                for off, val in txn.v_tblw.items():
                    self.tbl[off] = val
                self.seq += 1
                logger.debug(f"{sim.now} TXN {txn.id} CASOK   {self.tbl}")
                return True
            return False

    def try_ML_APPEND(self, sim, tbl_id: int, expected_offset: int, txn_id: int) -> bool:
        """Attempt physical append to a table's manifest list.

        In ML+ mode, entries are written tentatively (tagged with txn_id).
        Readers filter entries based on whether the associated transaction
        committed. Validity of the entry is determined by catalog commit outcome,
        not by table version at append time.

        Args:
            sim: SimPy environment
            tbl_id: Table ID
            expected_offset: Expected manifest list offset
            txn_id: Transaction ID (for tagging the entry)

        Returns:
            bool: True if physical append succeeded, False if offset moved or sealed
        """
        # Check if manifest list is sealed (needs rewrite)
        if self.ml_sealed[tbl_id]:
            logger.debug(f"{sim.now} ML_APPEND table {tbl_id} SEALED - needs rewrite")
            return False

        # Physical check: Does offset match?
        if self.ml_offset[tbl_id] != expected_offset:
            logger.debug(f"{sim.now} ML_APPEND table {tbl_id} PHYSICAL_FAIL - "
                        f"offset {self.ml_offset[tbl_id]} != expected {expected_offset}")
            return False

        # Physical success - entry written (tentative, tagged with txn_id)
        # Entry validity determined by catalog commit outcome, not table version
        self.ml_offset[tbl_id] += MANIFEST_LIST_ENTRY_SIZE
        self._check_ml_seal_threshold(tbl_id)
        logger.debug(f"{sim.now} ML_APPEND table {tbl_id} txn {txn_id} OK - offset now {self.ml_offset[tbl_id]}")
        return True

    def _check_ml_seal_threshold(self, tbl_id: int):
        """Check if manifest list threshold reached and seal if needed."""
        if MANIFEST_LIST_SEAL_THRESHOLD > 0 and self.ml_offset[tbl_id] >= MANIFEST_LIST_SEAL_THRESHOLD:
            self.ml_sealed[tbl_id] = True
            logger.debug(f"ML table {tbl_id} SEALED - threshold reached")

    def rewrite_manifest_list(self, tbl_id: int):
        """Rewrite manifest list (unseals it)."""
        self.ml_sealed[tbl_id] = False
        # Offset resets to entry size (one entry in new object)
        self.ml_offset[tbl_id] = MANIFEST_LIST_ENTRY_SIZE
        logger.debug(f"ML table {tbl_id} REWRITTEN - offset reset")


@dataclass
class Txn:
    id: int
    t_submit: int # ms submitted since start
    t_runtime: int # ms between submission and commit
    v_catalog_seq: int # version of catalog read (CAS in storage)
    v_tblr: dict[int, int] # versions of tables read
    v_tblw: dict[int, int] # versions of tables written
    n_retries: int = 0 # number of retries
    t_commit: int = field(default=-1)
    t_abort: int = field(default=-1)
    v_dirty: dict[int, int] = field(default_factory=lambda: defaultdict(dict)) # versions validated (init union(v_tblr, v_tblw))
    # Append mode fields
    v_log_offset: int = 0  # Log offset when snapshot was taken (for append mode)
    v_ml_offset: dict[int, int] = field(default_factory=dict)  # Per-table manifest list offsets


@dataclass
class LogEntry:
    """Log entry for append-based catalog operations.

    Each entry represents a committed transaction's effect on the catalog.
    Used for conflict detection and compaction in append mode.
    """
    txn_id: int  # Transaction ID (used for deduplication)
    tables_written: dict[int, int]  # table_id -> new_version after this txn
    tables_read: dict[int, int]  # table_id -> version_read by this txn
    sealed: bool = False  # True if this entry triggers compaction


class AppendCatalog:
    """Catalog implementation using append-based commit protocol.

    Instead of CAS for every commit, transactions append intention records
    to the catalog log. Validation happens at append time (like CAS, but
    at table-level instead of catalog-level).

    Key differences from CAS-based Catalog:
    - Conflict detection is table-level, not catalog-level
    - Concurrent non-conflicting transactions can both succeed
    - Physical failure (offset moved) just requires retry at new offset
    - Compaction required periodically to prevent unbounded log growth

    The simulation does NOT store log entries - it tracks:
    - log_offset: current byte position (for physical conflict detection)
    - tbl: per-table versions (updated on successful append)
    - committed_txn: set of successful transaction IDs
    """

    def __init__(self, sim):
        self.sim = sim
        self.seq = 0  # For compatibility with Txn class (tracks number of commits)
        self.tbl = [0] * N_TABLES  # Per-table versions
        self.log_offset = 0  # Current log byte offset
        self.checkpoint_offset = 0  # Offset of last checkpoint
        self.entries_since_checkpoint = 0  # Number of entries since last compaction
        self.sealed = False  # If True, next commit must CAS (compaction needed)
        self.committed_txn: set[int] = set()  # Successful transaction IDs
        # Per-table manifest list state for append mode
        self.ml_offset = [0] * N_TABLES  # Manifest list byte offset per table
        self.ml_sealed = [False] * N_TABLES  # Whether manifest list is sealed (needs rewrite)

    def try_APPEND(self, sim, txn, entry: LogEntry) -> tuple[bool, bool]:
        """Attempt conditional append with table-level validation.

        This models the append protocol where:
        1. Physical check: Does offset match? If not, retry at new offset.
        2. Logical check: Do table versions match? If not, conflict - must repair.

        The key insight: physical conflict rate is an upper bound on logical
        conflict rate. Physical failures are cheap (just retry), logical
        failures require reading table metadata/manifest to repair.

        Args:
            sim: SimPy environment
            txn: Transaction attempting to commit
            entry: Intention record to append

        Returns:
            (physical_success, logical_success):
            - (False, None): Offset moved, retry at catalog.log_offset
            - (True, False): Offset matched but table versions conflict
            - (True, True): Commit succeeded
        """
        logger.debug(f"{sim.now} TXN {txn.id} APPEND at offset {self.log_offset} "
                    f"(txn expected {txn.v_log_offset})")

        # Physical check: Does offset match?
        if self.log_offset != txn.v_log_offset:
            # Physical failure - offset moved, new offset returned to caller
            logger.debug(f"{sim.now} TXN {txn.id} APPEND_PHYSICAL_FAIL - "
                        f"offset moved to {self.log_offset}")
            return (False, None)

        # Physical success - now do logical validation (same as CAS but table-level)
        # Check that table versions match what we read/wrote
        for tbl_id, new_ver in entry.tables_written.items():
            expected_ver = new_ver - 1  # Writing v+1 expects current to be v
            if self.tbl[tbl_id] != expected_ver:
                # Logical conflict - table was modified
                logger.debug(f"{sim.now} TXN {txn.id} APPEND_LOGICAL_FAIL - "
                            f"table {tbl_id} at v{self.tbl[tbl_id]}, expected v{expected_ver}")
                # Still advance log offset (intention record is in log, just not applied)
                self.log_offset += LOG_ENTRY_SIZE
                self.entries_since_checkpoint += 1
                self._check_compaction_threshold(entry)
                return (True, False)

        # Check read-set for serializable isolation
        for tbl_id, read_ver in entry.tables_read.items():
            if tbl_id not in entry.tables_written:
                if self.tbl[tbl_id] != read_ver:
                    logger.debug(f"{sim.now} TXN {txn.id} APPEND_LOGICAL_FAIL - "
                                f"read table {tbl_id} at v{read_ver}, now v{self.tbl[tbl_id]}")
                    self.log_offset += LOG_ENTRY_SIZE
                    self.entries_since_checkpoint += 1
                    self._check_compaction_threshold(entry)
                    return (True, False)

        # Logical success - commit the transaction
        # Update table versions (inlined metadata means catalog merge yields table state)
        for tbl_id, new_ver in entry.tables_written.items():
            self.tbl[tbl_id] = new_ver

        self.committed_txn.add(entry.txn_id)
        self.seq += 1
        self.log_offset += LOG_ENTRY_SIZE
        self.entries_since_checkpoint += 1

        self._check_compaction_threshold(entry)

        logger.debug(f"{sim.now} TXN {txn.id} APPEND_OK - committed, offset now {self.log_offset}")
        return (True, True)

    def _check_compaction_threshold(self, entry: LogEntry):
        """Check if compaction threshold reached and seal if needed."""
        size_exceeded = (self.log_offset - self.checkpoint_offset) > COMPACTION_THRESHOLD
        entries_exceeded = (COMPACTION_MAX_ENTRIES > 0 and
                          self.entries_since_checkpoint >= COMPACTION_MAX_ENTRIES)

        if size_exceeded or entries_exceeded:
            entry.sealed = True
            self.sealed = True
            reason = "size" if size_exceeded else "entry count"
            logger.debug(f"SEALED - compaction needed ({reason})")
            STATS.append_compactions_triggered += 1

    def try_CAS_compact(self, sim, txn) -> bool:
        """Perform compaction via CAS.

        Replaces the log with a new checkpoint containing current state.

        Returns:
            True if compaction succeeded, False otherwise
        """
        logger.debug(f"{sim.now} TXN {txn.id} CAS_COMPACT")

        # In simulation, compaction always succeeds (we model contention via latency)
        self.checkpoint_offset = self.log_offset
        self.entries_since_checkpoint = 0
        self.sealed = False

        STATS.append_compactions_completed += 1
        logger.debug(f"{sim.now} TXN {txn.id} COMPACTED - offset {self.log_offset}")
        return True

    def try_ML_APPEND(self, sim, tbl_id: int, expected_offset: int, txn_id: int) -> bool:
        """Attempt physical append to a table's manifest list.

        See Catalog.try_ML_APPEND for full documentation.
        """
        # Check if manifest list is sealed (needs rewrite)
        if self.ml_sealed[tbl_id]:
            logger.debug(f"{sim.now} ML_APPEND table {tbl_id} SEALED - needs rewrite")
            return False

        # Physical check: Does offset match?
        if self.ml_offset[tbl_id] != expected_offset:
            logger.debug(f"{sim.now} ML_APPEND table {tbl_id} PHYSICAL_FAIL - "
                        f"offset {self.ml_offset[tbl_id]} != expected {expected_offset}")
            return False

        # Physical success - entry written (tentative, tagged with txn_id)
        self.ml_offset[tbl_id] += MANIFEST_LIST_ENTRY_SIZE
        self._check_ml_seal_threshold(tbl_id)
        logger.debug(f"{sim.now} ML_APPEND table {tbl_id} txn {txn_id} OK - offset now {self.ml_offset[tbl_id]}")
        return True

    def _check_ml_seal_threshold(self, tbl_id: int):
        """Check if manifest list threshold reached and seal if needed."""
        if MANIFEST_LIST_SEAL_THRESHOLD > 0 and self.ml_offset[tbl_id] >= MANIFEST_LIST_SEAL_THRESHOLD:
            self.ml_sealed[tbl_id] = True
            logger.debug(f"ML table {tbl_id} SEALED - threshold reached")

    def rewrite_manifest_list(self, tbl_id: int):
        """Rewrite manifest list (unseals it)."""
        self.ml_sealed[tbl_id] = False
        self.ml_offset[tbl_id] = MANIFEST_LIST_ENTRY_SIZE
        logger.debug(f"ML table {tbl_id} REWRITTEN - offset reset")


def txn_ml_w(sim, txn):
    """Write manifest lists for all tables written in transaction."""
    # Write each manifest list with latency from normal distribution
    for _ in txn.v_tblw:
        yield sim.timeout(get_manifest_list_latency('write'))
    logger.debug(f"{sim.now} TXN {txn.id} ML_W")


def txn_ml_append(sim, txn, catalog):
    """Append tentative entries to manifest lists (ML+ mode).

    In ML+ mode:
    1. Entries are written tentatively, tagged with txn_id
    2. Readers filter entries based on whether transaction committed
    3. Entry validity determined by CATALOG commit outcome, not ML append

    Physical append only - no logical validation at ML level. On catalog
    conflict, the conflict resolution logic determines whether ML entries
    need to be updated (real conflict) or are still valid (false conflict).

    Args:
        sim: SimPy environment
        txn: Transaction being committed
        catalog: Catalog (either Catalog or AppendCatalog)

    Yields:
        SimPy timeout events for I/O operations
    """
    for tbl_id in txn.v_tblw.keys():
        expected_offset = txn.v_ml_offset.get(tbl_id, 0)

        # Check if manifest list is sealed - must rewrite
        if catalog.ml_sealed[tbl_id]:
            STATS.manifest_append_sealed_rewrite += 1
            logger.debug(f"{sim.now} TXN {txn.id} ML table {tbl_id} SEALED - rewriting")

            # Read current manifest list, rewrite it
            yield sim.timeout(get_manifest_list_latency('read'))
            yield sim.timeout(get_manifest_list_latency('write'))
            catalog.rewrite_manifest_list(tbl_id)

            # Update our offset (entry is in the new object)
            txn.v_ml_offset[tbl_id] = catalog.ml_offset[tbl_id]
            STATS.manifest_append_physical_success += 1
            continue

        # Attempt physical append (tentative entry tagged with txn.id)
        yield sim.timeout(get_append_latency())
        physical_success = catalog.try_ML_APPEND(sim, tbl_id, expected_offset, txn.id)

        # Physical failure: offset moved or sealed, retry at new offset
        while not physical_success:
            STATS.manifest_append_physical_failure += 1

            # Check if sealed during our attempt
            if catalog.ml_sealed[tbl_id]:
                STATS.manifest_append_sealed_rewrite += 1
                logger.debug(f"{sim.now} TXN {txn.id} ML table {tbl_id} SEALED during append - rewriting")
                yield sim.timeout(get_manifest_list_latency('read'))
                yield sim.timeout(get_manifest_list_latency('write'))
                catalog.rewrite_manifest_list(tbl_id)
                txn.v_ml_offset[tbl_id] = catalog.ml_offset[tbl_id]
                break

            # Update offset and retry at new position
            txn.v_ml_offset[tbl_id] = catalog.ml_offset[tbl_id]
            yield sim.timeout(get_append_latency())
            physical_success = catalog.try_ML_APPEND(sim, tbl_id, txn.v_ml_offset[tbl_id], txn.id)

        STATS.manifest_append_physical_success += 1
        logger.debug(f"{sim.now} TXN {txn.id} ML_APPEND table {tbl_id} TENTATIVE")

    logger.debug(f"{sim.now} TXN {txn.id} ML_APPEND complete (entries tentative until catalog commit)")


def txn_commit(sim, txn, catalog):
    """Attempt to commit transaction with conflict resolution.

    This function orchestrates the commit process:
    1. Attempts CAS operation
    2. On success, commits the transaction
    3. On failure, uses ConflictResolver to:
       - Read manifest lists for missed snapshots
       - Merge table-level conflicts
       - Update transaction state for retry
    """
    # Attempt CAS operation
    yield sim.timeout(get_cas_latency())

    if catalog.try_CAS(sim, txn):
        # Success - transaction committed
        logger.debug(f"{sim.now} TXN {txn.id} commit")
        txn.t_commit = sim.now
        STATS.commit(txn)
    else:
        # CAS failed - need to resolve conflicts
        resolver = ConflictResolver()
        n_snapshots_behind = resolver.calculate_snapshots_behind(txn, catalog)
        logger.debug(f"{sim.now} TXN {txn.id} CAS Fail - {n_snapshots_behind} snapshots behind")

        if txn.n_retries >= N_TXN_RETRY:
            txn.t_abort = sim.now
            STATS.abort(txn)
            return

        # Read catalog to get current sequence number
        yield sim.timeout(get_cas_latency() / 2)

        # Update to current catalog state
        v_catalog = dict()
        txn.v_catalog_seq = catalog.seq
        for t in txn.v_dirty.keys():
            v_catalog[t] = catalog.tbl[t]

        # Read manifest lists for all snapshots between our read and current
        yield from resolver.read_manifest_lists(sim, n_snapshots_behind, txn.id)

        # Merge conflicts for affected tables
        # In ML+ mode, catalog is needed to re-append on real conflicts
        yield from resolver.merge_table_conflicts(sim, txn, v_catalog, catalog)

        # Update write set to the next available version per table
        resolver.update_write_set(txn, v_catalog)


def txn_commit_append(sim, txn, catalog: AppendCatalog):
    """Attempt to commit transaction using append-based protocol.

    Key insight: Physical conflict rate is an upper bound on logical conflict rate.
    - Physical failure (offset moved): Just retry at new offset (returned by failed append)
    - After physical success: Must re-read catalog to discover logical outcome
    - Logical failure (table conflict): Must repair - read manifest list, resolve conflicts

    The intention record contains inlined table metadata, so catalog merge yields
    table state directly without separate storage trip.
    """
    # If catalog is sealed, must perform compaction first
    if catalog.sealed:
        yield sim.timeout(get_compaction_latency())
        catalog.try_CAS_compact(sim, txn)

    # Create intention record for this transaction
    entry = LogEntry(
        txn_id=txn.id,
        tables_written=dict(txn.v_tblw),
        tables_read=dict(txn.v_tblr),
        sealed=False
    )

    # Attempt append - may need to retry at new offset on physical failure
    yield sim.timeout(get_append_latency())
    physical_success, logical_success = catalog.try_APPEND(sim, txn, entry)

    # Physical failure: offset moved, retry at new offset (no I/O needed)
    while not physical_success:
        STATS.append_physical_failure += 1

        # Update our offset to current (returned by failed append)
        txn.v_log_offset = catalog.log_offset

        # Retry append at new offset
        yield sim.timeout(get_append_latency())
        physical_success, logical_success = catalog.try_APPEND(sim, txn, entry)

    STATS.append_physical_success += 1

    # Physical success - but transaction doesn't know logical outcome yet
    # Must re-read catalog to discover if intention record was applied
    # The simulator computed logical_success, but we model the I/O cost
    yield sim.timeout(get_cas_latency() / 2)  # Read catalog to discover outcome

    if logical_success:
        # Logical success - transaction committed
        STATS.append_logical_success += 1
        txn.t_commit = sim.now
        STATS.commit(txn)
        logger.debug(f"{sim.now} TXN {txn.id} APPEND_COMMIT")
    else:
        # Logical failure - table version conflict discovered on re-read
        # Must repair: read manifest list, resolve conflicts
        STATS.append_logical_conflict += 1
        logger.debug(f"{sim.now} TXN {txn.id} APPEND_LOGICAL_CONFLICT")

        if txn.n_retries >= N_TXN_RETRY:
            txn.t_abort = sim.now
            STATS.abort(txn)
            return

        # Table metadata is inlined in catalog - already read above
        # Update transaction state from catalog (merged state)
        txn.v_log_offset = catalog.log_offset
        v_catalog = {}
        for t in txn.v_dirty.keys():
            v_catalog[t] = catalog.tbl[t]
            txn.v_dirty[t] = catalog.tbl[t]

        # Resolve conflict: read manifest list(s) for affected tables
        # This is same as CAS conflict resolution but at table level
        resolver = ConflictResolver()
        yield from resolver.read_manifest_lists(sim, 1, txn.id)  # Read current manifest

        # Check if conflict is real or false (same as CAS mode)
        # In ML+ mode:
        # - False conflict: ML entry still valid, no ML update needed
        # - Real conflict: ML entry needs update, re-append with merged data
        yield from resolver.merge_table_conflicts(sim, txn, v_catalog, catalog)

        # Update write set to next version
        resolver.update_write_set(txn, v_catalog)


def rand_tbl(catalog):
    """Select tables for transaction, respecting group boundaries."""
    # If no grouping (N_GROUPS == 1), use all tables
    if N_GROUPS == 1:
        available_tables = list(range(N_TABLES))
        table_pmf = TBL_R_PMF
        max_tables = N_TABLES
    else:
        # Select a random group
        group_id = np.random.choice(N_GROUPS)
        available_tables = GROUP_TO_TABLES[group_id]
        max_tables = len(available_tables)

        # Create PMF for tables in this group
        # Normalize the original PMF over the available tables
        table_pmf = np.array([TBL_R_PMF[t] for t in available_tables])
        table_pmf = table_pmf / table_pmf.sum()

    # how many tables (limit to group size)
    ntbl_requested = int(np.random.choice(np.arange(1, N_TABLES + 1), p=N_TBL_PMF))
    ntbl = min(ntbl_requested, max_tables)

    # Note: When num_groups == num_tables (table-level conflicts), groups have 1 table each.
    # Transactions requesting multiple tables are automatically capped to group size.
    # This is expected behavior, not an error.

    # which tables read
    tblr_idx = np.random.choice(available_tables, size=ntbl, replace=False, p=table_pmf).astype(int).tolist()
    tblr = {t: catalog.tbl[t] for t in tblr_idx}
    tblr_idx.sort()

    # write \subseteq read (not empty, read-only txn snapshot)
    ntblw = int(np.random.choice(np.arange(1, ntbl + 1), p=N_TBL_W_PMF[ntbl]))
    # uniform random from #tables to write
    tblw_idx = np.random.choice(tblr_idx, size=ntblw, replace=False).astype(int).tolist()
    # write versions = catalog versions + 1
    tblw = {t: tblr[t] + 1 for t in tblw_idx}
    return tblr, tblw

def txn_gen(sim, txn_id, catalog):
    tblr, tblw = rand_tbl(catalog)
    t_runtime = T_MIN_RUNTIME + np.random.lognormal(mean=T_RUNTIME_MU, sigma=T_RUNTIME_SIGMA)
    logger.debug(f"{sim.now} TXN {txn_id} {t_runtime} r {tblr} w {tblw}")
    # check all versions read/written TODO: serializable vs snapshot
    txn = Txn(txn_id, sim.now, int(t_runtime), catalog.seq, tblr, tblw)
    txn.v_dirty = txn.v_tblr.copy()
    txn.v_dirty.update(txn.v_tblw)

    # For append mode, also capture log offset
    if CATALOG_MODE == "append":
        txn.v_log_offset = catalog.log_offset

    # Capture manifest list offsets for append mode
    if MANIFEST_LIST_MODE == "append":
        for tbl_id in list(tblr.keys()) + list(tblw.keys()):
            txn.v_ml_offset[tbl_id] = catalog.ml_offset[tbl_id]

    # If table metadata is NOT inlined, read it separately
    if not TABLE_METADATA_INLINED:
        for tbl_id in list(tblr.keys()) + list(tblw.keys()):
            yield sim.timeout(get_table_metadata_latency('read'))

    # run the transaction
    yield sim.timeout(txn.t_runtime)

    # If table metadata is NOT inlined, write it before manifest list
    if not TABLE_METADATA_INLINED:
        for tbl_id in tblw.keys():
            yield sim.timeout(get_table_metadata_latency('write'))

    # write the manifest list (use append mode if configured)
    if MANIFEST_LIST_MODE == "append":
        yield sim.process(txn_ml_append(sim, txn, catalog))
    else:
        yield sim.process(txn_ml_w(sim, txn))
    while txn.t_commit < 0 and txn.t_abort < 0:
        # attempt commit
        txn.n_retries += 1

        # Apply exponential backoff before retry (if this is a retry, not first attempt)
        if txn.n_retries > 1:
            backoff_time = calculate_backoff_time(txn.n_retries - 1)
            if backoff_time > 0:
                logger.debug(f"{sim.now} TXN {txn_id} backing off for {backoff_time:.1f}ms (retry {txn.n_retries})")
                yield sim.timeout(backoff_time)

        # Use appropriate commit function based on catalog mode
        if CATALOG_MODE == "append":
            yield sim.process(txn_commit_append(sim, txn, catalog))
        else:
            yield sim.process(txn_commit(sim, txn, catalog))


def setup(sim):
    # Create appropriate catalog based on mode
    if CATALOG_MODE == "append":
        catalog = AppendCatalog(sim)
    else:
        catalog = Catalog(sim)

    txn_ids = itertools.count(1)
    sim.process(txn_gen(sim, next(txn_ids), catalog))
    while True:
        yield sim.timeout(int(generate_inter_arrival_time()))
        sim.process(txn_gen(sim, next(txn_ids), catalog))


def check_existing_experiment(config: dict, config_file: str) -> tuple[bool, int | None, str | None]:
    """Check if experiment results already exist and are complete.

    Returns:
        (skip, seed, output_path):
            - skip: True if results exist and simulation should be skipped
            - seed: Existing seed if found, None otherwise
            - output_path: Path to existing results if found, None otherwise
    """
    from pathlib import Path
    import tomllib

    label = config.get("experiment", {}).get("label")

    if label is None:
        # No experiment label - check if simple output file exists
        output_path = config["simulation"]["output_path"]
        if Path(output_path).exists():
            return (True, None, output_path)
        return (False, None, None)

    # Compute experiment hash
    exp_hash = compute_experiment_hash(config)
    exp_dir = Path("experiments") / f"{label}-{exp_hash}"

    # Check if experiment directory exists
    if not exp_dir.exists():
        return (False, None, None)

    # Check for existing cfg.toml and validate hash
    exp_config_path = exp_dir / "cfg.toml"
    if exp_config_path.exists():
        try:
            with open(exp_config_path, "rb") as f:
                existing_config = tomllib.load(f)
            existing_hash = compute_experiment_hash(existing_config)

            if existing_hash != exp_hash:
                logger.warning(f"⚠️  Hash mismatch in {exp_dir}")
                logger.warning(f"   Expected: {exp_hash}")
                logger.warning(f"   Found:    {existing_hash}")
                logger.warning(f"   Configuration may have changed - continuing anyway")
        except Exception as e:
            logger.warning(f"Could not validate existing config: {e}")

    # Look for completed runs (seed directories with results.parquet)
    output_filename = config["simulation"]["output_path"]
    completed_seeds = []

    if exp_dir.exists():
        for seed_dir in exp_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.isdigit():
                results_path = seed_dir / output_filename
                if results_path.exists():
                    completed_seeds.append(int(seed_dir.name))

    # Check if configured seed is already completed
    configured_seed = config.get("simulation", {}).get("seed")
    if configured_seed is not None and configured_seed in completed_seeds:
        output_path = exp_dir / str(configured_seed) / output_filename
        return (True, configured_seed, str(output_path))

    return (False, None, None)


def prepare_experiment_output(config: dict, config_file: str, actual_seed: int) -> str:
    """Prepare experiment output directory and return final output path.

    If experiment.label is set:
    - Creates: experiments/$label-$hash/$seed/
    - Writes: experiments/$label-$hash/cfg.toml (copy of input config)
    - Returns: experiments/$label-$hash/$seed/results.parquet

    If experiment.label is not set:
    - Returns: original output_path from config

    Args:
        config: Parsed TOML configuration dict
        config_file: Path to original config file
        actual_seed: The actual seed being used (either from config or randomly generated)

    Returns:
        Final output path for results
    """
    from pathlib import Path
    import shutil

    label = config.get("experiment", {}).get("label")

    if label is None:
        # No experiment label - use original output path
        return config["simulation"]["output_path"]

    # Compute experiment hash
    exp_hash = compute_experiment_hash(config)

    # Use actual seed (never "noseed" - always the real seed used)
    seed_str = str(actual_seed)

    # Construct paths
    exp_dir = Path("experiments") / f"{label}-{exp_hash}"
    run_dir = exp_dir / seed_str
    output_path = run_dir / config["simulation"]["output_path"]

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to experiment directory (if not already there)
    exp_config_path = exp_dir / "cfg.toml"
    if not exp_config_path.exists():
        shutil.copy2(config_file, exp_config_path)
        logger.info(f"Wrote experiment config to {exp_config_path}")

    logger.info(f"Experiment: {label}-{exp_hash}")
    logger.info(f"  Config hash: {exp_hash}")
    logger.info(f"  Seed: {seed_str}")
    logger.info(f"  Output: {output_path}")

    return str(output_path)


def cli():
    """CLI entry point for endive simulator."""
    parser = argparse.ArgumentParser(
        description="Iceberg-style catalog simulator for exploring commit latency tradeoffs"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="cfg.toml",
        help="Path to TOML configuration file (default: cfg.toml)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all logging except errors"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Load and validate configuration
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Validate configuration
    validation_errors = validate_config(config)
    if validation_errors:
        print("Configuration validation failed:")
        for error in validation_errors:
            print(f"  ✗ {error}")
        sys.exit(1)

    configure_from_toml(args.config)

    # Check if experiment results already exist
    skip, existing_seed, existing_output = check_existing_experiment(config, args.config)
    if skip:
        if not args.quiet:
            print(f"✓ Results already exist: {existing_output}")
            print(f"  Skipping simulation (seed={existing_seed})")
        sys.exit(0)

    # Print configuration (unless quiet mode)
    if not args.quiet:
        print_configuration()

    # Confirmation prompt (unless --yes or --quiet)
    if not args.yes and not args.quiet:
        if not confirm_run():
            print("Simulation cancelled.")
            sys.exit(0)

    # Setup random seed
    if SIM_SEED is not None:
        seed = SIM_SEED
        logger.info(f"Using configured seed: {seed}")
    else:
        seed = np.random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    np.random.seed(seed)

    # Prepare experiment output directory with actual seed
    final_output_path = prepare_experiment_output(config, args.config, seed)

    # Partition tables into groups (after seed is set for determinism)
    global TABLE_TO_GROUP, GROUP_TO_TABLES
    TABLE_TO_GROUP, GROUP_TO_TABLES = partition_tables_into_groups(
        N_TABLES, N_GROUPS, GROUP_SIZE_DIST, LONGTAIL_PARAMS
    )

    # Print group size information if using multiple groups
    if N_GROUPS > 1 and not args.quiet:
        group_sizes = [len(tables) for tables in GROUP_TO_TABLES.values()]
        print(f"[Group Partitioning]")
        print(f"  Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, "
              f"mean={sum(group_sizes)/len(group_sizes):.1f}")
        print()

    # Run simulation with progress bar
    logger.info(f"Starting simulation...")
    env = simpy.Environment()
    env.process(setup(env))

    # Progress bar setup
    show_progress = not args.no_progress and not args.verbose and not args.quiet
    if show_progress:
        with tqdm(total=SIM_DURATION_MS, unit='ms', unit_scale=True,
                  desc="Simulating", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            last_time = 0
            while env.peek() < SIM_DURATION_MS:
                next_time = min(env.peek() + SIM_DURATION_MS // 100, SIM_DURATION_MS)
                env.run(until=next_time)
                delta = env.now - last_time
                pbar.update(delta)
                last_time = env.now
            # Final update
            if env.now < SIM_DURATION_MS:
                env.run(until=SIM_DURATION_MS)
                pbar.update(SIM_DURATION_MS - last_time)
    else:
        env.run(until=SIM_DURATION_MS)

    logger.info("Simulation complete")

    # Export results to temporary file first, then rename to final path
    # This allows distinguishing complete runs from interrupted/partial runs
    from pathlib import Path
    import shutil

    output_dir = Path(final_output_path).parent
    temp_output_path = output_dir / ".running.parquet"

    logger.info(f"Exporting results to temporary file {temp_output_path}")
    STATS.export_parquet(str(temp_output_path))

    # Rename to final output path on success
    logger.info(f"Moving results to {final_output_path}")
    shutil.move(str(temp_output_path), final_output_path)
    logger.info(f"Results exported successfully")

if __name__ == "__main__":
    cli()
