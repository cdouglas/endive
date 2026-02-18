#!/usr/bin/env python

import argparse
import itertools
import logging
import os
import simpy
import subprocess
import sys
import tomllib
import numpy as np
from tqdm import tqdm
from endive.capstats import Stats, truncated_zipf_pmf, lognormal_params_from_mean_and_sigma
from endive.utils import get_git_sha, get_git_sha_short, partition_tables_into_groups
from endive.config import (
    PROVIDER_PROFILES,
    lognormal_mu_from_median,
    convert_mean_stddev_to_lognormal,
    parse_latency_config,
    get_provider_latency_config,
    apply_provider_defaults,
    compute_experiment_hash,
    validate_config,
)
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

# Partition-level modeling (within a single table)
# When enabled, transactions span multiple partitions with per-partition conflict detection
PARTITION_ENABLED: bool  # Enable partition-level modeling (default: False)
N_PARTITIONS: int  # Number of partitions per table
PARTITION_SELECTION_DIST: str  # How transactions select partitions: "zipf" or "uniform"
PARTITION_ZIPF_ALPHA: float  # Zipf exponent for hot partitions (higher = more skewed)
PARTITIONS_PER_TXN_MEAN: float  # Mean number of partitions per transaction
PARTITIONS_PER_TXN_MAX: int  # Maximum partitions per transaction

# Size-based PUT latency (Durner et al. VLDB 2023)
# latency = base_latency + size_mib * latency_per_mib + lognormal_noise
T_PUT: dict  # {'base_latency_ms': float, 'latency_per_mib_ms': float, 'sigma': float}
TABLE_METADATA_SIZE_BYTES: int  # Size of table metadata JSON file (~1-10 KB)
PARTITION_METADATA_ENTRY_SIZE: int  # Bytes per partition in table metadata (~100 bytes)

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



# Provider profiles, latency parsing, and validation moved to endive/config.py



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
    global T_PUT, TABLE_METADATA_SIZE_BYTES, PARTITION_METADATA_ENTRY_SIZE
    global PARTITION_ENABLED, N_PARTITIONS, PARTITION_SELECTION_DIST, PARTITION_ZIPF_ALPHA
    global PARTITIONS_PER_TXN_MEAN, PARTITIONS_PER_TXN_MAX

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

    # === PARTITION CONFIGURATION ===
    # Partition-level modeling: transactions span multiple partitions with per-partition conflicts
    partition_cfg = config.get("partition", {})
    PARTITION_ENABLED = partition_cfg.get("enabled", False)
    N_PARTITIONS = partition_cfg.get("num_partitions", 100)
    selection_cfg = partition_cfg.get("selection", {})
    PARTITION_SELECTION_DIST = selection_cfg.get("distribution", "zipf")
    PARTITION_ZIPF_ALPHA = selection_cfg.get("zipf_alpha", 1.5)
    PARTITIONS_PER_TXN_MEAN = partition_cfg.get("partitions_per_txn_mean", 3.0)
    PARTITIONS_PER_TXN_MAX = partition_cfg.get("partitions_per_txn_max", 10)

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

    # === SIZE-BASED PUT LATENCY (Durner et al. VLDB 2023) ===

    # T_PUT: size-based latency model for file writes
    # latency = base_latency + size_mib * latency_per_mib + lognormal_noise
    if 'T_PUT' in provider_defaults and 'T_PUT' not in storage_cfg:
        T_PUT = provider_defaults['T_PUT']
    elif 'T_PUT' in storage_cfg:
        T_PUT = storage_cfg['T_PUT']
    else:
        T_PUT = None  # Will fall back to legacy fixed latencies

    # TABLE_METADATA_SIZE_BYTES: Size of table metadata JSON file
    # Typical sizes: 1-10 KB for simple tables, up to 100 KB for complex schemas
    TABLE_METADATA_SIZE_BYTES = storage_cfg.get('table_metadata_size_bytes', 10 * 1024)  # Default 10 KB

    # PARTITION_METADATA_ENTRY_SIZE: Additional bytes per partition in table metadata
    # When partition.enabled=true, table metadata tracks N partition ML pointers
    # Typical: ~100-200 bytes per partition (pointer, offset, stats)
    PARTITION_METADATA_ENTRY_SIZE = storage_cfg.get('partition_metadata_entry_size', 100)

    # NOTE: Table partitioning is done in CLI after seed is set for determinism


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


def get_put_latency(size_bytes: int) -> float:
    """Get PUT operation latency based on file size.

    Uses the size-based latency model from Durner et al. VLDB 2023:
        latency = base_latency + size_mib * latency_per_mib + lognormal_noise

    The model accounts for:
    - Fixed connection/first-byte overhead (base_latency)
    - Data transfer time proportional to size (latency_per_mib)
    - Variance in cloud storage performance (lognormal noise)

    Args:
        size_bytes: Size of the file to write in bytes

    Returns:
        Latency in milliseconds.
    """
    if T_PUT is None:
        # Fallback to manifest list write latency if PUT not configured
        return get_manifest_list_latency('write')

    base_latency = T_PUT.get('base_latency_ms', 30)
    latency_per_mib = T_PUT.get('latency_per_mib_ms', 20)
    sigma = T_PUT.get('sigma', 0.3)

    # Convert bytes to MiB
    size_mib = size_bytes / (1024 * 1024)

    # Compute deterministic component: base + size * rate
    deterministic_latency = base_latency + size_mib * latency_per_mib

    # Add lognormal noise around the deterministic value
    # We use deterministic_latency as the median (exp(mu) = deterministic_latency)
    if deterministic_latency > 0:
        mu = np.log(deterministic_latency)
        latency = np.random.lognormal(mean=mu, sigma=sigma)
    else:
        latency = deterministic_latency

    return max(latency, MIN_LATENCY)


def get_table_metadata_size() -> int:
    """Get table metadata size in bytes, accounting for partitions.

    When partition mode is enabled, table metadata must track N partition
    manifest list pointers, adding O(N) overhead to metadata size.

    Returns:
        Size in bytes.
    """
    base_size = TABLE_METADATA_SIZE_BYTES
    if PARTITION_ENABLED:
        # Each partition adds an entry (ML pointer, offset, stats)
        partition_overhead = N_PARTITIONS * PARTITION_METADATA_ENTRY_SIZE
        return base_size + partition_overhead
    return base_size


def get_table_metadata_latency(operation: str) -> float:
    """Get table metadata read/write latency based on file size.

    For writes, uses the size-based PUT latency model.
    For reads, uses the same model (GET latency similar to PUT for small files).

    When partition mode is enabled, metadata size scales with N_PARTITIONS,
    creating an O(N) bottleneck even when only M partitions conflict.

    Args:
        operation: "read" or "write"

    Returns:
        Latency in milliseconds.
    """
    metadata_size = get_table_metadata_size()

    if T_PUT is not None:
        # Use size-based PUT latency
        return get_put_latency(metadata_size)
    else:
        # Fallback to legacy config (doesn't account for partition scaling)
        if operation == 'read':
            return generate_latency_from_config(T_TABLE_METADATA_R)
        else:
            return generate_latency_from_config(T_TABLE_METADATA_W)


def get_manifest_list_write_latency(size_bytes: int) -> float:
    """Get manifest list write latency based on current size.

    Uses the size-based PUT latency model for manifest list writes.

    Args:
        size_bytes: Current size of the manifest list in bytes

    Returns:
        Latency in milliseconds.
    """
    if T_PUT is not None:
        return get_put_latency(size_bytes)
    else:
        # Fallback to fixed latency
        return get_manifest_list_latency('write')


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

    def format_latency(config: dict) -> str:
        """Format latency config for display."""
        if not isinstance(config, dict):
            return str(config)
        if 'mu' in config:
            # Processed lognormal distribution: compute median from mu
            mu = float(config['mu'])
            sigma = config.get('sigma', 0)
            median = np.exp(mu)
            return f"{median:.1f}ms (σ={sigma:.2f})"
        elif 'median' in config:
            # Raw lognormal distribution: show median (sigma)
            return f"{config['median']:.1f}ms (σ={config['sigma']:.2f})"
        elif 'mean' in config:
            # Normal distribution: show mean±stddev
            return f"{config['mean']:.1f}±{config.get('stddev', 0):.1f}ms"
        else:
            return str(config)

    print("\n[Storage Latencies]")
    print(f"  Provider:     {STORAGE_PROVIDER}")
    print(f"  Min Latency:  {MIN_LATENCY:.1f}ms")
    print(f"  CAS:          {format_latency(T_CAS)}")
    if T_APPEND:
        print(f"  Append:       {format_latency(T_APPEND)}")
    print(f"  Metadata R:   {format_latency(T_METADATA_ROOT.get('read', T_METADATA_ROOT))}")
    print(f"  Metadata W:   {format_latency(T_METADATA_ROOT.get('write', T_METADATA_ROOT))}")
    print(f"  Manifest L R: {format_latency(T_MANIFEST_LIST.get('read', T_MANIFEST_LIST))}")
    print(f"  Manifest L W: {format_latency(T_MANIFEST_LIST.get('write', T_MANIFEST_LIST))}")
    print(f"  Manifest F R: {format_latency(T_MANIFEST_FILE.get('read', T_MANIFEST_FILE))}")
    print(f"  Manifest F W: {format_latency(T_MANIFEST_FILE.get('write', T_MANIFEST_FILE))}")
    print(f"  Max Parallel: {MAX_PARALLEL}")

    # Size-based PUT latency (Durner et al. VLDB 2023)
    if T_PUT is not None:
        print("\n[Size-Based Latency (Durner et al. VLDB 2023)]")
        print(f"  Base latency:     {T_PUT.get('base_latency_ms', 30):.1f}ms")
        print(f"  Latency/MiB:      {T_PUT.get('latency_per_mib_ms', 20):.1f}ms")
        print(f"  Variance (sigma): {T_PUT.get('sigma', 0.3):.2f}")
        print(f"  Table metadata:   {TABLE_METADATA_SIZE_BYTES/1024:.1f} KB")

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
    def read_manifest_lists(sim, n_snapshots: int, txn_id: int, avg_ml_size: int = None):
        """Read n manifest lists in batches respecting MAX_PARALLEL limit.

        Yields timeout events for simulating parallel I/O with batching.

        Args:
            sim: SimPy environment
            n_snapshots: Number of manifest lists to read
            txn_id: Transaction ID for logging
            avg_ml_size: Optional average manifest list size for size-based latency
        """
        if n_snapshots <= 0:
            return

        logger.debug(f"{sim.now} TXN {txn_id} Reading {n_snapshots} manifest lists (max_parallel={MAX_PARALLEL})")

        # Process in batches of MAX_PARALLEL
        for batch_start in range(0, n_snapshots, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_snapshots - batch_start)
            # All reads in this batch happen in parallel, take max time
            if T_PUT is not None and avg_ml_size is not None:
                # Use size-based latency (read ~= PUT for small files)
                batch_latencies = [get_put_latency(avg_ml_size) for _ in range(batch_size)]
            else:
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
                    # False conflict: Same table modified, but no data overlap (different partitions).
                    # Still requires creating a new snapshot with merged manifest list pointers.
                    # In ML+ mode: Tentative entry is still valid, no ML update needed.
                    # In traditional mode: Must read ML and write new ML with combined pointers.
                    yield from ConflictResolver.resolve_false_conflict(sim, txn, t, v_catalog, catalog)
                    STATS.false_conflicts += 1

    @staticmethod
    def resolve_false_conflict(sim, txn, table_id: int, v_catalog: dict, catalog=None):
        """Resolve false conflict (same table modified, no data overlap).

        A false conflict occurs when another transaction committed changes to the
        same table, but to different partitions (no overlapping data). The transaction
        must still create a new snapshot that includes both sets of changes.

        Required operations:
        - Read manifest list (to get committed snapshot's manifest file pointers)
        - Write NEW manifest list (combining both transactions' manifest file pointers)
        - Update table metadata (pointing to new manifest list)

        No manifest FILE operations are required (data doesn't overlap).

        In ML+ mode: The tentative ML entry (appended before CAS attempt) is still
        valid because it references this transaction's manifest files. Readers filter
        it until commit. No new ML write needed - just update table metadata pointer.

        In traditional (rewrite) mode: Must read the committed snapshot's manifest
        list and write a new one that includes both sets of manifest file pointers.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            v_catalog: Current catalog state
            catalog: Catalog reference (needed for size-based latency)
        """
        logger.debug(f"{sim.now} TXN {txn.id} Resolving false conflict for table {table_id}")

        # Read metadata root to understand new snapshot
        yield sim.timeout(get_metadata_root_latency('read'))

        # Read table metadata (JSON blob) to get current state
        # This is required even for false conflicts to merge snapshot metadata
        if not TABLE_METADATA_INLINED:
            yield sim.timeout(get_table_metadata_latency('read'))

        # Manifest list operations depend on mode
        if MANIFEST_LIST_MODE == "append":
            # ML+ mode: Tentative entry is still valid (different partition, no data overlap).
            # Readers filter entries by committed transaction list.
            # No ML update needed - the entry will become visible when CAS succeeds.
            logger.debug(f"{sim.now} TXN {txn.id} ML+ mode: tentative entry still valid, no ML update")
        else:
            # Traditional (rewrite) mode: Must create new manifest list combining
            # the committed snapshot's manifest file pointers with our own.

            # Read manifest list (to get pointers to committed snapshot's manifest files)
            if catalog is not None and T_PUT is not None:
                ml_size = catalog.ml_offset[table_id]
                yield sim.timeout(get_put_latency(ml_size))  # Read ~= PUT for small files
            else:
                yield sim.timeout(get_manifest_list_latency('read'))
            STATS.manifest_list_reads += 1

            # Write new manifest list (combined pointers from both transactions)
            if catalog is not None and T_PUT is not None:
                ml_size = catalog.ml_offset[table_id]
                yield sim.timeout(get_manifest_list_write_latency(ml_size))
            else:
                yield sim.timeout(get_manifest_list_latency('write'))
            STATS.manifest_list_writes += 1

            logger.debug(f"{sim.now} TXN {txn.id} Rewrite mode: wrote new manifest list")

        # Write merged table metadata with our snapshot included
        if not TABLE_METADATA_INLINED:
            yield sim.timeout(get_table_metadata_latency('write'))

        # Update validation version
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

        # Read table metadata (JSON blob) to get current snapshot state
        if not TABLE_METADATA_INLINED:
            yield sim.timeout(get_table_metadata_latency('read'))

        # Read manifest list (to get pointers to manifest files)
        # Use size-based latency if catalog and T_PUT available
        if catalog is not None and T_PUT is not None:
            ml_size = catalog.ml_offset[table_id]
            yield sim.timeout(get_put_latency(ml_size))  # Read ~= PUT for small files
        else:
            yield sim.timeout(get_manifest_list_latency('read'))
        STATS.manifest_list_reads += 1

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
                    # Sealed - rewrite (size-based latency)
                    STATS.manifest_append_sealed_rewrite += 1
                    ml_size = catalog.ml_offset[table_id]
                    if T_PUT is not None:
                        yield sim.timeout(get_put_latency(ml_size))  # Read
                        yield sim.timeout(get_manifest_list_write_latency(ml_size))  # Write
                    else:
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
            # Traditional mode: Write updated manifest list (size-based latency)
            if catalog is not None and T_PUT is not None:
                ml_size = catalog.ml_offset[table_id]
                yield sim.timeout(get_manifest_list_write_latency(ml_size))
            else:
                yield sim.timeout(get_manifest_list_latency('write'))
            STATS.manifest_list_writes += 1

        # Write merged table metadata (JSON blob) with updated manifest list pointer
        if not TABLE_METADATA_INLINED:
            yield sim.timeout(get_table_metadata_latency('write'))

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

    # === Partition-level conflict resolution ===

    @staticmethod
    def calculate_partition_snapshots_behind(txn: 'Txn', catalog, table_id: int, partition_id: int) -> int:
        """Calculate how many snapshots a partition is behind.

        Like table mode, Iceberg validates conflicts by traversing all snapshots
        between the transaction's read and the current state. For partition mode,
        this is per-partition: each partition has its own version counter.

        Returns: n where current partition is at version n and transaction read version m,
                 so transaction is (n - m) snapshots behind for this partition.
        """
        current_version = catalog.partition_seq[table_id][partition_id]
        txn_version = txn.v_partition_seq.get(table_id, {}).get(partition_id, 0)
        return current_version - txn_version

    @staticmethod
    def read_partition_manifest_lists(sim, n_snapshots: int, txn_id: int, table_id: int,
                                       partition_id: int, avg_ml_size: int = None):
        """Read n manifest lists for a partition's snapshot history.

        Like table mode's read_manifest_lists, but scoped to a single partition.
        Iceberg's validationHistory traverses all snapshots between starting and
        current, reading manifests from each to detect conflicts.

        Args:
            sim: SimPy environment
            n_snapshots: Number of manifest lists to read
            txn_id: Transaction ID for logging
            table_id: Table ID
            partition_id: Partition ID
            avg_ml_size: Optional average manifest list size for size-based latency
        """
        if n_snapshots <= 0:
            return

        logger.debug(f"{sim.now} TXN {txn_id} Reading {n_snapshots} partition MLs "
                    f"for table {table_id} partition {partition_id}")

        # Process in batches of MAX_PARALLEL
        for batch_start in range(0, n_snapshots, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_snapshots - batch_start)
            # All reads in this batch happen in parallel, take max time
            if T_PUT is not None and avg_ml_size is not None:
                batch_latencies = [get_put_latency(avg_ml_size) for _ in range(batch_size)]
            else:
                batch_latencies = [get_manifest_list_latency('read') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))

        STATS.manifest_list_reads += n_snapshots

    @staticmethod
    def merge_partition_conflicts(sim, txn: 'Txn', catalog):
        """Merge conflicts for partitions that have changed.

        In partition mode, conflicts are per-(table, partition). Like Iceberg's
        validationHistory, we must read manifest lists from ALL snapshots between
        the transaction's read and current state to properly detect conflicts.

        Args:
            sim: SimPy environment
            txn: Transaction object
            catalog: Catalog with current partition state
        """
        if not PARTITION_ENABLED:
            return

        # conflicting is dict[table_id, set[partition_id]]
        conflicting = catalog.get_conflicting_partitions(txn)
        if not conflicting:
            return

        total_conflicts = sum(len(parts) for parts in conflicting.values())
        logger.debug(f"{sim.now} TXN {txn.id} Resolving conflicts for {total_conflicts} partitions across {len(conflicting)} tables")

        for tbl_id, partitions in conflicting.items():
            for p in partitions:
                # Calculate how many snapshots behind for this partition
                n_behind = ConflictResolver.calculate_partition_snapshots_behind(txn, catalog, tbl_id, p)

                # Read manifest lists for all intermediate snapshots (like table mode)
                # This is required by Iceberg's validationHistory which traverses
                # all snapshots between starting and parent
                avg_ml_size = catalog.partition_ml_offset[tbl_id][p] if hasattr(catalog, 'partition_ml_offset') else None
                yield from ConflictResolver.read_partition_manifest_lists(
                    sim, n_behind, txn.id, tbl_id, p, avg_ml_size
                )

                # Determine if this is a real or false conflict
                is_real_conflict = np.random.random() < REAL_CONFLICT_PROBABILITY

                if is_real_conflict:
                    yield from ConflictResolver.resolve_partition_real_conflict(sim, txn, tbl_id, p, catalog)
                    STATS.real_conflicts += 1
                else:
                    yield from ConflictResolver.resolve_partition_false_conflict(sim, txn, tbl_id, p, catalog)
                    STATS.false_conflicts += 1

    @staticmethod
    def resolve_partition_false_conflict(sim, txn, table_id: int, partition_id: int, catalog):
        """Resolve false conflict for a table's partition (different data, no overlap).

        In partition mode with distributed MLs, each (table, partition) has its own ML.
        False conflict: Different partition ranges modified, no manifest file ops needed.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            partition_id: Partition with conflict
            catalog: Catalog reference
        """
        logger.debug(f"{sim.now} TXN {txn.id} False conflict table {table_id} partition {partition_id}")

        if MANIFEST_LIST_MODE == "append":
            # ML+ mode: Tentative entry is still valid, no ML update needed
            logger.debug(f"{sim.now} TXN {txn.id} ML+ mode: table {table_id} partition {partition_id} entry still valid")
        else:
            # Traditional mode: Read and rewrite partition's manifest list
            ml_size = catalog.partition_ml_offset[table_id][partition_id]
            if T_PUT is not None:
                yield sim.timeout(get_put_latency(ml_size))  # Read
                yield sim.timeout(get_manifest_list_write_latency(ml_size))  # Write
            else:
                yield sim.timeout(get_manifest_list_latency('read'))
                yield sim.timeout(get_manifest_list_latency('write'))
            STATS.manifest_list_reads += 1
            STATS.manifest_list_writes += 1

        # Update transaction's partition state
        txn.v_partition_seq[table_id][partition_id] = catalog.partition_seq[table_id][partition_id]
        txn.v_partition_ml_offset[table_id][partition_id] = catalog.partition_ml_offset[table_id][partition_id]

    @staticmethod
    def resolve_partition_real_conflict(sim, txn, table_id: int, partition_id: int, catalog):
        """Resolve real conflict for a table's partition (overlapping data).

        Real conflict: Same partition range modified, must read/write manifest files.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            partition_id: Partition with conflict
            catalog: Catalog reference
        """
        n_conflicting = sample_conflicting_manifests()

        logger.debug(f"{sim.now} TXN {txn.id} Real conflict table {table_id} partition {partition_id} "
                    f"({n_conflicting} manifests)")

        # Read partition's manifest list
        ml_size = catalog.partition_ml_offset[table_id][partition_id]
        if T_PUT is not None:
            yield sim.timeout(get_put_latency(ml_size))
        else:
            yield sim.timeout(get_manifest_list_latency('read'))
        STATS.manifest_list_reads += 1

        # Read conflicting manifest files
        for batch_start in range(0, n_conflicting, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
            batch_latencies = [get_manifest_file_latency('read') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))
        STATS.manifest_files_read += n_conflicting

        # Write merged manifest files
        for batch_start in range(0, n_conflicting, MAX_PARALLEL):
            batch_size = min(MAX_PARALLEL, n_conflicting - batch_start)
            batch_latencies = [get_manifest_file_latency('write') for _ in range(batch_size)]
            yield sim.timeout(max(batch_latencies))
        STATS.manifest_files_written += n_conflicting

        # Write updated partition ML
        if MANIFEST_LIST_MODE == "append" and hasattr(catalog, 'try_PARTITION_ML_APPEND'):
            expected_offset = txn.v_partition_ml_offset.get(table_id, {}).get(partition_id, 0)
            yield sim.timeout(get_append_latency())
            success = catalog.try_PARTITION_ML_APPEND(sim, table_id, partition_id, expected_offset, txn.id)

            while not success:
                STATS.manifest_append_physical_failure += 1
                if catalog.partition_ml_sealed[table_id][partition_id]:
                    STATS.manifest_append_sealed_rewrite += 1
                    ml_size = catalog.partition_ml_offset[table_id][partition_id]
                    if T_PUT is not None:
                        yield sim.timeout(get_put_latency(ml_size))
                        yield sim.timeout(get_manifest_list_write_latency(ml_size))
                    else:
                        yield sim.timeout(get_manifest_list_latency('write'))
                    catalog.rewrite_partition_manifest_list(table_id, partition_id)
                    txn.v_partition_ml_offset[table_id][partition_id] = catalog.partition_ml_offset[table_id][partition_id]
                    break
                txn.v_partition_ml_offset[table_id][partition_id] = catalog.partition_ml_offset[table_id][partition_id]
                yield sim.timeout(get_append_latency())
                success = catalog.try_PARTITION_ML_APPEND(sim, table_id, partition_id, txn.v_partition_ml_offset[table_id][partition_id], txn.id)

            STATS.manifest_append_physical_success += 1
        else:
            # Traditional mode: Write partition ML
            if T_PUT is not None:
                yield sim.timeout(get_manifest_list_write_latency(ml_size))
            else:
                yield sim.timeout(get_manifest_list_latency('write'))
            STATS.manifest_list_writes += 1

        # Update transaction's partition state
        txn.v_partition_seq[table_id][partition_id] = catalog.partition_seq[table_id][partition_id]
        txn.v_partition_ml_offset[table_id][partition_id] = catalog.partition_ml_offset[table_id][partition_id]

    @staticmethod
    def update_partition_state(txn: 'Txn', catalog):
        """Update transaction's partition state after conflict resolution.

        Refreshes the transaction's view of partition versions for retry.
        """
        if not PARTITION_ENABLED:
            return

        for tbl_id, partitions in txn.partitions_read.items():
            if tbl_id not in txn.v_partition_seq:
                txn.v_partition_seq[tbl_id] = {}
            if tbl_id not in txn.v_partition_ml_offset:
                txn.v_partition_ml_offset[tbl_id] = {}
            for p in partitions:
                txn.v_partition_seq[tbl_id][p] = catalog.partition_seq[tbl_id][p]
                txn.v_partition_ml_offset[tbl_id][p] = catalog.partition_ml_offset[tbl_id][p]

        for tbl_id, partitions in txn.partitions_written.items():
            if tbl_id not in txn.v_partition_seq:
                txn.v_partition_seq[tbl_id] = {}
            if tbl_id not in txn.v_partition_ml_offset:
                txn.v_partition_ml_offset[tbl_id] = {}
            for p in partitions:
                txn.v_partition_seq[tbl_id][p] = catalog.partition_seq[tbl_id][p]
                txn.v_partition_ml_offset[tbl_id][p] = catalog.partition_ml_offset[tbl_id][p]


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

        # Partition-level state (when PARTITION_ENABLED)
        # Each TABLE has its own set of partitions with version counters and manifest lists
        # Indexed as: partition_seq[table_id][partition_id]
        if PARTITION_ENABLED:
            self.partition_seq = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
            self.partition_ml_offset = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
            self.partition_ml_sealed = [[False] * N_PARTITIONS for _ in range(N_TABLES)]

    def try_CAS(self, sim, txn):
        """Attempt compare-and-swap for transaction commit.

        Conflict detection modes:
        1. Partition mode (PARTITION_ENABLED): Per-partition vector clock comparison
        2. Table-level (N_GROUPS == N_TABLES): Per-table version comparison
        3. Catalog-level (default): Global sequence number comparison
        """
        logger.debug(f"{sim.now} TXN {txn.id} CAS {self.seq} = {txn.v_catalog_seq} {txn.v_tblw}")
        logger.debug(f"{sim.now} TXN {txn.id} Catalog {self.tbl}")

        # Partition-level conflicts (vector clock comparison)
        if PARTITION_ENABLED:
            return self._try_CAS_partition(sim, txn)

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

    def _try_CAS_partition(self, sim, txn):
        """Partition-level CAS using vector clock comparison.

        Checks partitions PER TABLE that the transaction touched.
        Conflict occurs if any (table, partition) has a different version.
        """
        # Check if any touched (table, partition) has advanced
        # txn.partitions_read is dict[table_id, set[partition_id]]
        for tbl_id, partitions in txn.partitions_read.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    logger.debug(f"{sim.now} TXN {txn.id} CAS FAIL - table {tbl_id} partition {p} "
                                f"at v{self.partition_seq[tbl_id][p]}, expected v{txn.v_partition_seq[tbl_id][p]}")
                    return False

        for tbl_id, partitions in txn.partitions_written.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    logger.debug(f"{sim.now} TXN {txn.id} CAS FAIL - table {tbl_id} partition {p} "
                                f"at v{self.partition_seq[tbl_id][p]}, expected v{txn.v_partition_seq[tbl_id][p]}")
                    return False

        # No conflicts - commit: increment only written partitions
        for tbl_id, partitions in txn.partitions_written.items():
            for p in partitions:
                self.partition_seq[tbl_id][p] += 1
        self.seq += 1  # Global seq for total ordering/stats
        logger.debug(f"{sim.now} TXN {txn.id} CASOK (partition-level)")
        return True

    def get_conflicting_partitions(self, txn) -> dict[int, set[int]]:
        """Get the set of partitions per table that have conflicts.

        Returns:
            dict[table_id, set[partition_id]]: Conflicting partitions per table
        """
        if not PARTITION_ENABLED:
            return {}

        conflicting = {}
        # Check read partitions
        for tbl_id, partitions in txn.partitions_read.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    if tbl_id not in conflicting:
                        conflicting[tbl_id] = set()
                    conflicting[tbl_id].add(p)

        # Check written partitions
        for tbl_id, partitions in txn.partitions_written.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    if tbl_id not in conflicting:
                        conflicting[tbl_id] = set()
                    conflicting[tbl_id].add(p)

        return conflicting

    def try_PARTITION_ML_APPEND(self, sim, table_id: int, partition_id: int, expected_offset: int, txn_id: int) -> bool:
        """Attempt physical append to a table's partition manifest list.

        Similar to try_ML_APPEND but operates on per-(table, partition) manifest lists.
        """
        if not PARTITION_ENABLED:
            return False

        # Check if partition ML is sealed
        if self.partition_ml_sealed[table_id][partition_id]:
            logger.debug(f"{sim.now} PARTITION_ML_APPEND table {table_id} partition {partition_id} SEALED")
            return False

        # Physical check: Does offset match?
        if self.partition_ml_offset[table_id][partition_id] != expected_offset:
            logger.debug(f"{sim.now} PARTITION_ML_APPEND table {table_id} partition {partition_id} PHYSICAL_FAIL - "
                        f"offset {self.partition_ml_offset[table_id][partition_id]} != expected {expected_offset}")
            return False

        # Physical success
        self.partition_ml_offset[table_id][partition_id] += MANIFEST_LIST_ENTRY_SIZE
        self._check_partition_ml_seal_threshold(table_id, partition_id)
        logger.debug(f"{sim.now} PARTITION_ML_APPEND table {table_id} partition {partition_id} txn {txn_id} OK")
        return True

    def _check_partition_ml_seal_threshold(self, table_id: int, partition_id: int):
        """Check if partition ML threshold reached and seal if needed."""
        if (MANIFEST_LIST_SEAL_THRESHOLD > 0 and
            self.partition_ml_offset[table_id][partition_id] >= MANIFEST_LIST_SEAL_THRESHOLD):
            self.partition_ml_sealed[table_id][partition_id] = True
            logger.debug(f"Table {table_id} partition {partition_id} ML SEALED - threshold reached")

    def rewrite_partition_manifest_list(self, table_id: int, partition_id: int):
        """Rewrite partition manifest list (unseals it)."""
        if not PARTITION_ENABLED:
            return
        self.partition_ml_sealed[table_id][partition_id] = False
        self.partition_ml_offset[table_id][partition_id] = MANIFEST_LIST_ENTRY_SIZE
        logger.debug(f"Table {table_id} partition {partition_id} ML REWRITTEN")

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
    # Partition mode fields (when PARTITION_ENABLED)
    # Indexed by table_id -> set of partition_ids or table_id -> partition_id -> value
    partitions_read: dict[int, set[int]] = field(default_factory=dict)  # table_id -> partition_ids read
    partitions_written: dict[int, set[int]] = field(default_factory=dict)  # table_id -> partition_ids written
    v_partition_seq: dict[int, dict[int, int]] = field(default_factory=dict)  # table_id -> partition_id -> version
    v_partition_ml_offset: dict[int, dict[int, int]] = field(default_factory=dict)  # table_id -> partition_id -> offset


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

        # Partition-level state (when PARTITION_ENABLED)
        # Each TABLE has its own set of partitions
        if PARTITION_ENABLED:
            self.partition_seq = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
            self.partition_ml_offset = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
            self.partition_ml_sealed = [[False] * N_PARTITIONS for _ in range(N_TABLES)]

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

    # Partition-level methods (delegate to Catalog pattern)
    def get_conflicting_partitions(self, txn) -> dict[int, set[int]]:
        """Get the set of partitions per table that have conflicts."""
        if not PARTITION_ENABLED:
            return {}
        conflicting = {}
        for tbl_id, partitions in txn.partitions_read.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    if tbl_id not in conflicting:
                        conflicting[tbl_id] = set()
                    conflicting[tbl_id].add(p)
        for tbl_id, partitions in txn.partitions_written.items():
            for p in partitions:
                if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                    if tbl_id not in conflicting:
                        conflicting[tbl_id] = set()
                    conflicting[tbl_id].add(p)
        return conflicting

    def try_PARTITION_ML_APPEND(self, sim, table_id: int, partition_id: int, expected_offset: int, txn_id: int) -> bool:
        """Attempt physical append to a table's partition manifest list."""
        if not PARTITION_ENABLED:
            return False
        if self.partition_ml_sealed[table_id][partition_id]:
            return False
        if self.partition_ml_offset[table_id][partition_id] != expected_offset:
            return False
        self.partition_ml_offset[table_id][partition_id] += MANIFEST_LIST_ENTRY_SIZE
        if (MANIFEST_LIST_SEAL_THRESHOLD > 0 and
            self.partition_ml_offset[table_id][partition_id] >= MANIFEST_LIST_SEAL_THRESHOLD):
            self.partition_ml_sealed[table_id][partition_id] = True
        return True

    def rewrite_partition_manifest_list(self, table_id: int, partition_id: int):
        """Rewrite partition manifest list (unseals it)."""
        if not PARTITION_ENABLED:
            return
        self.partition_ml_sealed[table_id][partition_id] = False
        self.partition_ml_offset[table_id][partition_id] = MANIFEST_LIST_ENTRY_SIZE


def txn_ml_w(sim, txn, catalog=None):
    """Write manifest lists for all tables written in transaction.

    Uses size-based PUT latency when T_PUT is configured (Durner et al. VLDB 2023).
    For each table being written, the manifest list size is estimated from
    the catalog's ml_offset tracking.

    Args:
        sim: SimPy environment
        txn: Transaction being committed
        catalog: Optional catalog for size-based latency (uses fixed latency if None)
    """
    for tbl_id in txn.v_tblw:
        if catalog is not None and T_PUT is not None:
            # Use size-based latency from catalog's manifest list size
            ml_size = catalog.ml_offset[tbl_id] + MANIFEST_LIST_ENTRY_SIZE
            yield sim.timeout(get_manifest_list_write_latency(ml_size))
        else:
            # Fallback to fixed latency
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

            # Read current manifest list, rewrite it (size-based latency)
            ml_size = catalog.ml_offset[tbl_id]
            if T_PUT is not None:
                yield sim.timeout(get_put_latency(ml_size))  # Read ~= PUT for small files
                yield sim.timeout(get_manifest_list_write_latency(ml_size))
            else:
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
                ml_size = catalog.ml_offset[tbl_id]
                if T_PUT is not None:
                    yield sim.timeout(get_put_latency(ml_size))  # Read ~= PUT for small files
                    yield sim.timeout(get_manifest_list_write_latency(ml_size))
                else:
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
       - Read manifest lists for missed snapshots (table mode)
       - Or resolve per-partition conflicts (partition mode)
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

        if txn.n_retries >= N_TXN_RETRY:
            txn.t_abort = sim.now
            STATS.abort(txn)
            return

        # Read catalog to get current sequence number
        yield sim.timeout(get_cas_latency() / 2)

        if PARTITION_ENABLED:
            # Partition mode: Resolve per-partition conflicts
            logger.debug(f"{sim.now} TXN {txn.id} CAS Fail - resolving partition conflicts")

            # Update catalog seq and table state
            txn.v_catalog_seq = catalog.seq
            v_catalog = dict()
            for t in txn.v_dirty.keys():
                v_catalog[t] = catalog.tbl[t]

            # Read table metadata to discover partition states
            # Cost scales with N_PARTITIONS (O(N) bottleneck even for M conflicts)
            if not TABLE_METADATA_INLINED:
                yield sim.timeout(get_table_metadata_latency('read'))

            # Resolve per-partition conflicts (handles reading partition MLs)
            yield from resolver.merge_partition_conflicts(sim, txn, catalog)

            # Read current catalog state for retry
            # This is a separate round-trip to get the latest partition versions
            # Without this latency, information would "teleport" from catalog to client
            yield sim.timeout(get_cas_latency())

            # Now capture the current state (after the read completes)
            txn.v_catalog_seq = catalog.seq
            for t in txn.v_dirty.keys():
                v_catalog[t] = catalog.tbl[t]

            # Update partition state for retry
            resolver.update_partition_state(txn, catalog)
            resolver.update_write_set(txn, v_catalog)
        else:
            # Table mode: Original behavior
            n_snapshots_behind = resolver.calculate_snapshots_behind(txn, catalog)
            logger.debug(f"{sim.now} TXN {txn.id} CAS Fail - {n_snapshots_behind} snapshots behind")

            # Update to current catalog state
            v_catalog = dict()
            txn.v_catalog_seq = catalog.seq
            for t in txn.v_dirty.keys():
                v_catalog[t] = catalog.tbl[t]

            # Read manifest lists for all snapshots between our read and current
            # Use average manifest list size for size-based latency estimation
            avg_ml_size = sum(catalog.ml_offset) // len(catalog.ml_offset) if catalog.ml_offset else None
            yield from resolver.read_manifest_lists(sim, n_snapshots_behind, txn.id, avg_ml_size)

            # Merge conflicts for affected tables
            # In ML+ mode, catalog is needed to re-append on real conflicts
            yield from resolver.merge_table_conflicts(sim, txn, v_catalog, catalog)

            # Read current catalog state for retry
            # This is a separate round-trip to get the latest table versions
            # Without this latency, information would "teleport" from catalog to client
            yield sim.timeout(get_cas_latency())

            # Re-capture catalog state after the read completes
            txn.v_catalog_seq = catalog.seq
            for t in txn.v_dirty.keys():
                v_catalog[t] = catalog.tbl[t]

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
        # Use average manifest list size for size-based latency estimation
        avg_ml_size = sum(catalog.ml_offset) // len(catalog.ml_offset) if catalog.ml_offset else None
        yield from resolver.read_manifest_lists(sim, 1, txn.id, avg_ml_size)  # Read current manifest

        # Check if conflict is real or false (same as CAS mode)
        # In ML+ mode:
        # - False conflict: ML entry still valid, no ML update needed
        # - Real conflict: ML entry needs update, re-append with merged data
        yield from resolver.merge_table_conflicts(sim, txn, v_catalog, catalog)

        # Read current catalog state for retry
        # This is a separate round-trip to get the latest table versions
        yield sim.timeout(get_cas_latency())

        # Re-capture catalog state after the read completes
        txn.v_log_offset = catalog.log_offset
        for t in txn.v_dirty.keys():
            v_catalog[t] = catalog.tbl[t]

        # Update write set to next version
        resolver.update_write_set(txn, v_catalog)


def select_partitions(n_partitions: int) -> tuple[set[int], set[int]]:
    """Select partitions for transaction based on configured distribution.

    Returns:
        (partitions_read, partitions_written): Sets of partition IDs
    """
    # How many partitions to touch (geometric/exponential around mean, capped)
    # Also cap at n_partitions to avoid sampling errors
    n_parts = max(1, min(
        int(np.random.exponential(PARTITIONS_PER_TXN_MEAN)),
        PARTITIONS_PER_TXN_MAX,
        n_partitions  # Can't select more partitions than exist
    ))

    # Select partitions based on distribution
    if PARTITION_SELECTION_DIST == "zipf":
        # Zipf distribution: partition 0 is hottest
        # Use truncated Zipf PMF
        pmf = truncated_zipf_pmf(n_partitions, PARTITION_ZIPF_ALPHA)
        partitions = np.random.choice(
            n_partitions, size=n_parts, replace=False, p=pmf
        ).astype(int).tolist()
    else:  # uniform
        partitions = np.random.choice(
            n_partitions, size=n_parts, replace=False
        ).astype(int).tolist()

    partitions_read = set(partitions)

    # Write subset of read (at least 1)
    n_write = max(1, int(np.random.exponential(PARTITIONS_PER_TXN_MEAN / 2)))
    n_write = min(n_write, len(partitions))
    partitions_written = set(np.random.choice(
        list(partitions_read), size=n_write, replace=False
    ).astype(int).tolist())

    return partitions_read, partitions_written


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

    # Partition mode: select partitions for EACH table and capture per-partition state
    if PARTITION_ENABLED:
        # For each table read, select partitions
        txn.partitions_read = {}
        txn.partitions_written = {}
        txn.v_partition_seq = {}
        txn.v_partition_ml_offset = {}

        for tbl_id in tblr.keys():
            parts_r, parts_w = select_partitions(N_PARTITIONS)
            txn.partitions_read[tbl_id] = parts_r
            # Only include in written if table is being written
            if tbl_id in tblw:
                txn.partitions_written[tbl_id] = parts_w
            # Capture per-partition versions (vector clock snapshot)
            txn.v_partition_seq[tbl_id] = {}
            txn.v_partition_ml_offset[tbl_id] = {}
            for p in parts_r:
                txn.v_partition_seq[tbl_id][p] = catalog.partition_seq[tbl_id][p]
                txn.v_partition_ml_offset[tbl_id][p] = catalog.partition_ml_offset[tbl_id][p]

        logger.debug(f"{sim.now} TXN {txn_id} partitions r={txn.partitions_read} w={txn.partitions_written}")

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
        yield sim.process(txn_ml_w(sim, txn, catalog))
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

    # Write version info (git SHA) for reproducibility
    version_path = exp_dir / "version.txt"
    git_sha = get_git_sha()
    if not version_path.exists():
        with open(version_path, 'w') as f:
            f.write(f"git_sha={git_sha}\n")
        logger.info(f"Wrote version info to {version_path} (git_sha={git_sha[:7]})")

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
    validation_errors, validation_warnings = validate_config(config)
    if validation_warnings:
        print("Configuration warnings:")
        for warning in validation_warnings:
            print(f"  ⚠ {warning}")
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
