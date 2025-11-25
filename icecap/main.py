#!/usr/bin/env python

import argparse
import itertools
import logging
import simpy
import sys
import tomllib
import numpy as np
from tqdm import tqdm
from icecap.capstats import Stats, truncated_zipf_pmf, lognormal_params_from_mean_and_sigma
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
# Storage operation latencies (normal distributions with mean and stddev)
T_CAS: dict  # {'mean': float, 'stddev': float}
T_METADATA_ROOT: dict  # {'read': {'mean': float, 'stddev': float}, 'write': {...}}
T_MANIFEST_LIST: dict
T_MANIFEST_FILE: dict
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

STATS = Stats()

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
    2. Hash of simulator code files (icecap/*.py)

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
    icecap_dir = Path(__file__).parent

    # Include all .py files in icecap/ directory
    for py_file in sorted(icecap_dir.glob("*.py")):
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

    # Load storage latencies (normal distributions)
    MAX_PARALLEL = config["storage"]["max_parallel"]
    MIN_LATENCY = config["storage"]["min_latency"]

    T_CAS = {
        'mean': config["storage"]["T_CAS"]["mean"],
        'stddev': config["storage"]["T_CAS"]["stddev"]
    }

    T_METADATA_ROOT = {
        'read': {
            'mean': config["storage"]["T_METADATA_ROOT"]["read"]["mean"],
            'stddev': config["storage"]["T_METADATA_ROOT"]["read"]["stddev"]
        },
        'write': {
            'mean': config["storage"]["T_METADATA_ROOT"]["write"]["mean"],
            'stddev': config["storage"]["T_METADATA_ROOT"]["write"]["stddev"]
        }
    }

    T_MANIFEST_LIST = {
        'read': {
            'mean': config["storage"]["T_MANIFEST_LIST"]["read"]["mean"],
            'stddev': config["storage"]["T_MANIFEST_LIST"]["read"]["stddev"]
        },
        'write': {
            'mean': config["storage"]["T_MANIFEST_LIST"]["write"]["mean"],
            'stddev': config["storage"]["T_MANIFEST_LIST"]["write"]["stddev"]
        }
    }

    T_MANIFEST_FILE = {
        'read': {
            'mean': config["storage"]["T_MANIFEST_FILE"]["read"]["mean"],
            'stddev': config["storage"]["T_MANIFEST_FILE"]["read"]["stddev"]
        },
        'write': {
            'mean': config["storage"]["T_MANIFEST_FILE"]["write"]["mean"],
            'stddev': config["storage"]["T_MANIFEST_FILE"]["write"]["stddev"]
        }
    }

    # Load runtime-related configuration
    N_TXN_RETRY = config["transaction"]["retry"]
    T_MIN_RUNTIME = config["transaction"]["runtime"]["min"]
    mean = config["transaction"]["runtime"]["mean"]
    sigma = config["transaction"]["runtime"]["sigma"]
    T_RUNTIME_MU, T_RUNTIME_SIGMA = lognormal_params_from_mean_and_sigma(mean, sigma)

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
    """Generate storage operation latency from normal distribution.

    Enforces minimum latency to prevent unrealistic zero or near-zero values.
    """
    return max(MIN_LATENCY, np.random.normal(loc=mean, scale=stddev))


def get_cas_latency() -> float:
    """Get CAS operation latency."""
    return generate_latency(T_CAS['mean'], T_CAS['stddev'])


def get_metadata_root_latency(operation: str) -> float:
    """Get metadata root latency for read or write operation."""
    params = T_METADATA_ROOT[operation]
    return generate_latency(params['mean'], params['stddev'])


def get_manifest_list_latency(operation: str) -> float:
    """Get manifest list latency for read or write operation."""
    params = T_MANIFEST_LIST[operation]
    return generate_latency(params['mean'], params['stddev'])


def get_manifest_file_latency(operation: str) -> float:
    """Get manifest file latency for read or write operation."""
    params = T_MANIFEST_FILE[operation]
    return generate_latency(params['mean'], params['stddev'])


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
    print("  ICECAP SIMULATOR CONFIGURATION")
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
    def merge_table_conflicts(sim, txn: 'Txn', v_catalog: dict):
        """Merge conflicts for tables that have changed.

        For each dirty table that has a different version in the catalog,
        determines if this is a false conflict (version changed but no data overlap)
        or a real conflict (overlapping data changes), and resolves accordingly.

        Yields timeout events for each I/O operation.
        """
        for t, v in txn.v_dirty.items():
            if v_catalog[t] != v:
                # Determine if this is a real conflict
                is_real_conflict = np.random.random() < REAL_CONFLICT_PROBABILITY

                if is_real_conflict:
                    yield from ConflictResolver.resolve_real_conflict(sim, txn, t, v_catalog)
                    STATS.real_conflicts += 1
                else:
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
    def resolve_real_conflict(sim, txn, table_id: int, v_catalog: dict):
        """Resolve real conflict (overlapping data changes).

        Must read and rewrite manifest files to merge conflicting changes.
        The number of conflicting manifests is sampled from configuration.

        Args:
            sim: SimPy environment
            txn: Transaction object
            table_id: Table with conflict
            v_catalog: Current catalog state
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

        # Write updated manifest list
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

def txn_ml_w(sim, txn):
    """Write manifest lists for all tables written in transaction."""
    # Write each manifest list with latency from normal distribution
    for _ in txn.v_tblw:
        yield sim.timeout(get_manifest_list_latency('write'))
    logger.debug(f"{sim.now} TXN {txn.id} ML_W")

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
        yield from resolver.merge_table_conflicts(sim, txn, v_catalog)

        # Update write set to the next available version per table
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
    # run the transaction
    yield sim.timeout(txn.t_runtime)
    # write the manifest list
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

        yield sim.process(txn_commit(sim, txn, catalog))

def setup(sim):
    catalog = Catalog(sim)
    txn_ids = itertools.count(1)
    sim.process(txn_gen(sim, next(txn_ids), catalog))
    while True:
        yield sim.timeout(int(generate_inter_arrival_time()))
        sim.process(txn_gen(sim, next(txn_ids), catalog))


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
    """CLI entry point for icecap simulator."""
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
