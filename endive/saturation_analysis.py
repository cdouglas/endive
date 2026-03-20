#!/usr/bin/env python
"""
Saturation analysis module for baseline experiments.

This module generates the latency vs throughput graphs described in ANALYSIS_PLAN.md.
It reads experiment directories, extracts parameters from cfg.toml files,
aggregates results across multiple seeds, and produces visualization.
"""

import argparse
import fnmatch
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomli

# Global configuration (loaded from analysis.toml or defaults)
CONFIG = {}

# Default DPI for all plots (12in × 100dpi = 1200px wide)
DEFAULT_DPI = 100


def get_default_config() -> Dict:
    """Return default configuration if no config file is provided."""
    return {
        'paths': {
            'input_dir': 'experiments',
            'output_dir': 'plots',
            'pattern': 'exp2_*',
            'consolidated_file': 'experiments/consolidated.parquet'
        },
        'analysis': {
            'group_by': None,
            'k_min_cycles': 5,
            'min_warmup_ms': 300000,
            'max_warmup_ms': 900000,
            'min_seeds': 3,
            'use_consolidated': True
        },
        'plots': {
            'dpi': DEFAULT_DPI,
            'bbox_inches': 'tight',
            'figsize': {
                'latency_vs_throughput': [12, 8],
                'success_vs_load': [12, 4.5],
                'success_vs_throughput': [12, 8],
                'overhead_vs_throughput': [12, 8]
            },
            'fonts': {
                'title': 16,
                'axis_label': 14,
                'legend': 10,
                'annotation': 10,
                'axis_value': 13
            },
            'font_weights': {
                'title': 'bold',
                'axis_label': 'bold'
            },
            'legend': {
                'loc': 'best',
                'framealpha': 0.9
            },
            'annotation': {
                'text_coords': 'offset points',
                'xy_text': [0, 10],
                'ha': 'center'
            },
            'saturation': {
                'enabled': False,  # Disabled by default (can be enabled in analysis.toml)
                'threshold': 50.0,
                'tolerance': 5.0
            },
            'stddev': {
                'enabled': True,
                'alpha': 0.2
            },
            'styles': {
                'markers': ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'],
                'linestyles': ['-', '--', '-.', ':'],
                'linewidth': 2,
                'markersize': 8,
                'alpha': 0.7
            },
            'grid': {
                'enabled': True,
                'alpha': 0.3
            },
            'percentiles': {
                'values': [50, 95, 99],
                'labels': ['P50', 'P95', 'P99']
            },
            'colors': {
                'scheme': 'default'
            }
        },
        'output': {
            'files': {
                'experiment_index': 'experiment_index.csv',
                'latency_vs_throughput_plot': 'latency_vs_throughput.png',
                'latency_vs_throughput_table': 'latency_vs_throughput.md',
                'success_vs_load_plot': 'success_vs_load.png',
                'success_vs_throughput_plot': 'success_vs_throughput.png',
                'overhead_vs_throughput_plot': 'overhead_vs_throughput.png',
                'overhead_vs_throughput_table': 'overhead_vs_throughput.md',
                'commit_rate_over_time_plot': 'commit_rate_over_time.png',
                'sustainable_throughput_plot': 'sustainable_throughput.png'
            },
            'table': {
                'float_format': '%.2f',
                'na_rep': 'N/A',
                'index': True
            }
        }
    }


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load analysis configuration from TOML file.

    Args:
        config_path: Path to analysis.toml file. If None, searches for analysis.toml
                     in current directory, then uses defaults.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try to find analysis.toml in current directory
        if os.path.exists('analysis.toml'):
            config_path = 'analysis.toml'
        else:
            # Use defaults
            return get_default_config()

    try:
        with open(config_path, 'rb') as f:
            loaded_config = tomli.load(f)

        # Merge with defaults (in case some keys are missing)
        default_config = get_default_config()

        # Deep merge
        def deep_merge(base, override):
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(default_config, loaded_config)

    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found, using defaults")
        return get_default_config()
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        print("Using default configuration")
        return get_default_config()


def scan_experiment_directories(base_dir: str, pattern: str) -> Dict[str, Dict]:
    """
    Scan experiment directories and build index of experiments.

    Returns:
        Dictionary mapping experiment_dir -> {
            'config': parsed config dict,
            'seeds': list of seed directories,
            'label': experiment label,
            'hash': experiment hash
        }
    """
    experiments = {}

    # Find all experiment directories matching pattern
    search_pattern = os.path.join(base_dir, pattern)
    exp_dirs = glob(search_pattern)

    for exp_dir in exp_dirs:
        # Check if cfg.toml exists
        cfg_path = os.path.join(exp_dir, "cfg.toml")
        if not os.path.exists(cfg_path):
            print(f"Warning: No cfg.toml in {exp_dir}, skipping")
            continue

        # Parse config
        with open(cfg_path, 'rb') as f:
            config = tomli.load(f)

        # Find all seed directories (numeric directory names)
        seed_dirs = []
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path) and item.isdigit():
                seed_dirs.append(item_path)

        if not seed_dirs:
            print(f"Warning: No seed directories in {exp_dir}, skipping")
            continue

        # Extract label and hash from directory name
        dir_name = os.path.basename(exp_dir)
        if '-' in dir_name:
            label, exp_hash = dir_name.rsplit('-', 1)
        else:
            label = dir_name
            exp_hash = "unknown"

        # Read code_hash from version.txt (if present)
        code_hash = 'unknown'
        version_path = os.path.join(exp_dir, 'version.txt')
        if os.path.exists(version_path):
            try:
                with open(version_path) as vf:
                    for line in vf:
                        if line.startswith('code_hash='):
                            code_hash = line.split('=', 1)[1].strip()
                            break
            except Exception:
                pass

        experiments[exp_dir] = {
            'config': config,
            'seeds': seed_dirs,
            'label': label,
            'hash': exp_hash,
            'dir': exp_dir,
            'code_hash': code_hash,
        }

    return experiments


def scan_consolidated_experiments(consolidated_path: str, pattern: str) -> Dict[str, Dict]:
    """Discover experiments from consolidated parquet when directories are absent.

    Returns the same dict structure as scan_experiment_directories(), keyed by
    ``"<exp_name>-<exp_hash>"``.  For experiments found only in the consolidated
    file the ``'dir'`` value is ``None`` and ``'seeds'`` is a list of int seed
    values (not directory paths).

    Uses row-group metadata to avoid reading any data for the discovery phase,
    then reads only one row per experiment for config reconstruction.
    """
    import pyarrow.parquet as pq

    if not os.path.exists(consolidated_path):
        return {}

    try:
        pf = pq.ParquetFile(consolidated_path)
    except Exception as e:
        print(f"Warning: Could not open consolidated file {consolidated_path}: {e}")
        return {}

    if pf.metadata.num_row_groups == 0:
        return {}

    # Phase 1: Scan row-group metadata (reads zero data bytes)
    schema = pf.schema_arrow
    try:
        name_idx = schema.get_field_index('exp_name')
        hash_idx = schema.get_field_index('exp_hash')
        seed_idx = schema.get_field_index('seed')
    except KeyError as e:
        print(f"Warning: Consolidated file missing expected column: {e}")
        return {}

    # code_hash column may not exist in older consolidated files
    try:
        code_hash_idx = schema.get_field_index('code_hash')
        if code_hash_idx < 0:
            code_hash_idx = None
    except (KeyError, ValueError):
        code_hash_idx = None

    # Collect (exp_name, exp_hash) -> {seeds, first_rg, code_hash} from row-group statistics
    groups: Dict[Tuple[str, str], Dict] = {}
    has_stats = True

    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        name_col = rg.column(name_idx)
        hash_col = rg.column(hash_idx)
        seed_col = rg.column(seed_idx)

        if not name_col.is_stats_set or not hash_col.is_stats_set or not seed_col.is_stats_set:
            has_stats = False
            break

        exp_name = name_col.statistics.min
        exp_hash = hash_col.statistics.min
        seed = seed_col.statistics.min

        # Read code_hash from row-group statistics if available
        rg_code_hash = 'unknown'
        if code_hash_idx is not None:
            ch_col = rg.column(code_hash_idx)
            if ch_col.is_stats_set:
                rg_code_hash = ch_col.statistics.min

        # Each row group should have homogeneous exp_name/exp_hash/seed
        # (consolidate.py writes one row group per experiment-seed)
        key = (exp_name, exp_hash)
        if key not in groups:
            groups[key] = {'seeds': set(), 'first_rg': i, 'code_hash': rg_code_hash}
        groups[key]['seeds'].add(seed)

    # Fallback: if statistics are absent, read only lightweight columns
    if not has_stats:
        # Determine which columns to read for fallback
        fallback_cols = ['exp_name', 'exp_hash', 'seed']
        if code_hash_idx is not None:
            fallback_cols.append('code_hash')

        try:
            table = pq.read_table(
                consolidated_path,
                columns=fallback_cols,
            )
        except Exception as e:
            print(f"Warning: Could not read consolidated file {consolidated_path}: {e}")
            return {}

        if table.num_rows == 0:
            return {}

        exp_names = table.column('exp_name').to_pylist()
        exp_hashes = table.column('exp_hash').to_pylist()
        seeds = table.column('seed').to_pylist()
        code_hashes = (table.column('code_hash').to_pylist()
                       if code_hash_idx is not None else ['unknown'] * len(exp_names))

        groups = {}
        for idx in range(len(exp_names)):
            key = (exp_names[idx], exp_hashes[idx])
            if key not in groups:
                groups[key] = {'seeds': set(), 'first_rg': None,
                               'code_hash': code_hashes[idx] or 'unknown'}
            groups[key]['seeds'].add(seeds[idx])

        # Map experiments to their first row group for config reads
        for i in range(pf.metadata.num_row_groups):
            rg_table = pf.read_row_group(i, columns=['exp_name', 'exp_hash'])
            if rg_table.num_rows > 0:
                rg_name = rg_table.column('exp_name')[0].as_py()
                rg_hash = rg_table.column('exp_hash')[0].as_py()
                rg_key = (rg_name, rg_hash)
                if rg_key in groups and groups[rg_key]['first_rg'] is None:
                    groups[rg_key]['first_rg'] = i

    # Phase 2: Filter by pattern + min_seeds, then read config per experiment
    experiments = {}
    min_seeds = CONFIG.get('analysis', {}).get('min_seeds', 3)

    for (exp_name, exp_hash), info in groups.items():
        dir_key = f"{exp_name}-{exp_hash}"
        if not fnmatch.fnmatch(dir_key, pattern):
            continue
        if len(info['seeds']) < min_seeds:
            continue

        # Read config from one row of the first matching row group
        rg_idx = info['first_rg']
        if rg_idx is None:
            continue
        try:
            rg_table = pf.read_row_group(rg_idx, columns=['config'])
            config = unflatten_config(rg_table.column('config')[0].as_py())
        except Exception as e:
            print(f"Warning: Could not read config for {dir_key}: {e}")
            continue

        experiments[dir_key] = {
            'config': config,
            'seeds': sorted(info['seeds']),
            'label': exp_name,
            'hash': exp_hash,
            'dir': None,  # no directory on disk
            'code_hash': info.get('code_hash', 'unknown'),
        }

    return experiments


def _filter_stale_experiments(experiments: Dict[str, Dict]) -> Dict[str, Dict]:
    """Remove stale experiment cohorts when multiple code versions exist per label.

    Groups experiments by label. If a label has multiple code_hash values,
    keeps only the cohort with the most experiments (ties broken by
    lexicographically-latest code_hash).
    """
    by_label: Dict[str, List] = defaultdict(list)
    for key, info in experiments.items():
        by_label[info['label']].append((key, info))

    filtered = {}
    for label, entries in by_label.items():
        code_hashes: Dict[str, List] = defaultdict(list)
        for key, info in entries:
            ch = info.get('code_hash', 'unknown')
            code_hashes[ch].append((key, info))

        if len(code_hashes) <= 1:
            for key, info in entries:
                filtered[key] = info
            continue

        # Multiple cohorts — keep the largest (most experiments), break ties alphabetically
        best_ch = max(code_hashes, key=lambda ch: (len(code_hashes[ch]), ch))
        stale_count = sum(len(v) for k, v in code_hashes.items() if k != best_ch)
        print(f"Warning: {label} has {len(code_hashes)} code versions "
              f"({stale_count} stale experiments removed, keeping code_hash={best_ch})")
        for key, info in code_hashes[best_ch]:
            filtered[key] = info

    return filtered


def scan_all_experiments(base_dir: str, pattern: str) -> Dict[str, Dict]:
    """Scan disk directories and consolidated parquet, merge with dedup.

    Returns dict keyed by ``"exp_name-exp_hash"``.  Disk experiments take
    precedence over consolidated when both exist for the same key.
    Stale experiment cohorts (old code versions) are filtered out.
    """
    experiments = {}

    # Phase 1: Scan disk directories (keyed by full path internally)
    for exp_dir_path, exp_info in scan_experiment_directories(base_dir, pattern).items():
        dir_key = os.path.basename(exp_dir_path)  # "exp_name-exp_hash"
        experiments[dir_key] = exp_info

    # Phase 2: Augment with consolidated-only experiments
    consolidated_path = os.path.join(base_dir, 'consolidated.parquet')
    if CONFIG.get('analysis', {}).get('use_consolidated', True) and os.path.exists(consolidated_path):
        for key, exp_info in scan_consolidated_experiments(consolidated_path, pattern).items():
            if key not in experiments:  # Both use "exp_name-exp_hash" keys now
                experiments[key] = exp_info

    # Phase 3: Filter stale cohorts (different code versions for same label)
    experiments = _filter_stale_experiments(experiments)

    return experiments


def compute_transient_period_duration(config: Dict) -> float:
    """
    Compute transient period duration using transaction-runtime multiple approach.

    Used for both warmup (start) and cooldown (end) periods to exclude transient
    behavior and focus on steady-state performance.

    The warmup period allows the system to reach steady-state by eliminating
    initial transient behavior (empty start, low contention, queue buildup).

    The cooldown period excludes end-of-simulation artifacts where the arrival
    process winds down, reducing contention and creating artificially low latencies.

    Approach: K_MIN_CYCLES × mean_transaction_runtime
    - Allows multiple transaction lifecycles for steady-state
    - Clamped to [5min, 15min] for practical bounds

    Args:
        config: Parsed TOML configuration dict

    Returns:
        Transient period duration in milliseconds
    """
    # Get config values
    K_MIN_CYCLES = CONFIG.get('analysis', {}).get('k_min_cycles', 5)
    MIN_PERIOD_MS = CONFIG.get('analysis', {}).get('min_warmup_ms', 300000)
    MAX_PERIOD_MS = CONFIG.get('analysis', {}).get('max_warmup_ms', 900000)

    # Get mean transaction runtime
    mean_runtime_ms = config.get("transaction", {}).get("runtime", {}).get("mean", 10000)

    # Calculate transient period based on transaction cycles
    period_ms = max(
        MIN_PERIOD_MS,
        min(
            K_MIN_CYCLES * mean_runtime_ms,
            MAX_PERIOD_MS
        )
    )

    return period_ms


def compute_warmup_duration(config: Dict) -> float:
    """
    Compute warmup duration (alias for compute_transient_period_duration).

    Returns:
        Warmup duration in milliseconds
    """
    return compute_transient_period_duration(config)


def compute_cooldown_duration(config: Dict) -> float:
    """
    Compute cooldown duration (alias for compute_transient_period_duration).

    Returns:
        Cooldown duration in milliseconds
    """
    return compute_transient_period_duration(config)


def _infer_type(value: str):
    """Infer Python type from a stringified config value."""
    if value in ('True', 'true'):
        return True
    if value in ('False', 'false'):
        return False
    try:
        int_val = int(value)
        # Only return int if round-trips exactly (avoid '3.0' -> 3)
        if str(int_val) == value:
            return int_val
    except (ValueError, OverflowError):
        pass
    try:
        return float(value)
    except (ValueError, OverflowError):
        pass
    return value


def unflatten_config(flat_config) -> Dict:
    """Reconstruct nested dict from flattened (key, value) pairs.

    Reverses flatten_config() used during consolidation.  Accepts any
    iterable of (key, value) tuples — including a PyArrow MapArray row.
    """
    result: Dict = {}
    for key, value in flat_config:
        # PyArrow may return pyarrow StringScalar; coerce to str
        key = str(key)
        value = str(value)
        parts = key.split('.')
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _infer_type(value)
    return result


def extract_key_parameters(config: Dict) -> Dict:
    """Extract key parameters from config for grouping/plotting."""
    params = {}

    # Extract inter-arrival time
    if 'transaction' in config and 'inter_arrival' in config['transaction']:
        ia = config['transaction']['inter_arrival']
        if isinstance(ia, dict):
            params['inter_arrival_scale'] = ia.get('scale', None)
        elif 'scale' in config['transaction']:
            params['inter_arrival_scale'] = config['transaction'].get('scale', None)

    # Extract catalog config
    if 'catalog' in config:
        params['num_tables'] = config['catalog'].get('num_tables', None)

    # Extract transaction config
    if 'transaction' in config:
        params['real_conflict_probability'] = config['transaction'].get('real_conflict_probability', 0.0)

        # Extract conflicting_manifests configuration
        if 'conflicting_manifests' in config['transaction']:
            cm = config['transaction']['conflicting_manifests']
            if isinstance(cm, dict):
                dist_type = cm.get('distribution', 'exponential')
                params['conflicting_manifests_distribution'] = dist_type

                # For fixed distribution, also extract the value
                if dist_type == 'fixed':
                    params['conflicting_manifests_value'] = cm.get('value', None)
                    # Create combined identifier for filtering (e.g., "fixed-1", "fixed-5")
                    if 'value' in cm:
                        params['conflicting_manifests_type'] = f"fixed-{cm['value']}"
                elif dist_type == 'exponential':
                    params['conflicting_manifests_mean'] = cm.get('mean', None)
                    params['conflicting_manifests_type'] = 'exponential'
                elif dist_type == 'uniform':
                    params['conflicting_manifests_type'] = 'uniform'

    # Extract simulation duration
    if 'simulation' in config:
        params['duration_ms'] = config['simulation'].get('duration_ms', None)

    # Extract storage/catalog latency parameters
    if 'storage' in config:
        # Extract storage provider (s3, s3x, azure, azurex, gcp, instant)
        params['storage_provider'] = config['storage'].get('provider', None)

        # Extract T_CAS (catalog CAS operation latency)
        if 'T_CAS' in config['storage']:
            t_cas = config['storage']['T_CAS']
            if isinstance(t_cas, dict):
                params['t_cas_mean'] = t_cas.get('mean', None)
                params['t_cas_median'] = t_cas.get('median', None)

    # Extract catalog mode and backend
    if 'catalog' in config:
        params['catalog_mode'] = config['catalog'].get('mode', 'cas')
        params['catalog_backend'] = config['catalog'].get('backend', 'storage')
        params['table_metadata_inlined'] = config['catalog'].get('table_metadata_inlined', True)

        # Extract catalog service latency (for catalog latency sweep experiments)
        if 'service' in config['catalog']:
            service = config['catalog']['service']
            if isinstance(service, dict):
                params['catalog_service_latency_ms'] = service.get('latency_ms', None)

    # Extract manifest list mode
    if 'transaction' in config:
        params['manifest_list_mode'] = config['transaction'].get('manifest_list_mode', 'rewrite')

    # Extract operation type parameters (for FA/VO mix experiments)
    if 'transaction' in config and 'operation_types' in config['transaction']:
        op_types = config['transaction']['operation_types']
        if isinstance(op_types, dict):
            params['fast_append_ratio'] = op_types.get('fast_append', None)
            params['validated_overwrite_ratio'] = op_types.get('validated_overwrite', None)

    # Extract table selection parameters
    if 'transaction' in config and 'table_selection' in config['transaction']:
        ts = config['transaction']['table_selection']
        if isinstance(ts, dict):
            params['table_selection_distribution'] = ts.get('distribution', 'uniform')
            if ts.get('distribution') == 'zipf':
                params['table_selection_zipf_alpha'] = ts.get('zipf_alpha', 1.5)

    # Extract partition parameters
    if 'partition' in config:
        params['partition_enabled'] = config['partition'].get('enabled', False)
        params['num_partitions'] = config['partition'].get('num_partitions', None)
        if 'partitions_per_txn' in config['partition']:
            ppt = config['partition']['partitions_per_txn']
            if isinstance(ppt, dict):
                params['partitions_per_txn_mean'] = ppt.get('mean', None)
                params['partitions_per_txn_max'] = ppt.get('max', None)

    return params


def load_and_aggregate_results(exp_info: Dict) -> pd.DataFrame:
    """
    Load all seed results for an experiment and aggregate statistics.

    Applies warmup and cooldown period filters to exclude transient behavior
    and focus on steady-state performance.

    Returns:
        DataFrame with aggregated statistics across all seeds (warmup/cooldown excluded)
    """
    # Compute warmup and cooldown durations for this experiment
    warmup_ms = compute_warmup_duration(exp_info['config'])
    cooldown_ms = compute_cooldown_duration(exp_info['config'])

    # Get simulation duration to compute cooldown threshold
    sim_duration_ms = exp_info['config'].get('simulation', {}).get('duration_ms', 3600000)
    cooldown_start_ms = sim_duration_ms - cooldown_ms

    all_results = []
    total_txns_before_filter = 0
    total_txns_after_filter = 0

    for seed_dir in exp_info['seeds']:
        # Find results.parquet in seed directory
        parquet_path = os.path.join(seed_dir, "results.parquet")

        if not os.path.exists(parquet_path):
            print(f"Warning: No results.parquet in {seed_dir}")
            continue

        # Load results
        df = pd.read_parquet(parquet_path)

        # Apply warmup and cooldown filters - only keep steady-state transactions
        total_txns_before_filter += len(df)
        df = df[(df['t_submit'] >= warmup_ms) & (df['t_submit'] < cooldown_start_ms)].copy()
        total_txns_after_filter += len(df)

        # Add seed identifier
        seed = os.path.basename(seed_dir)
        df['seed'] = seed

        all_results.append(df)

    if not all_results:
        return None

    # Combine all seeds
    combined = pd.concat(all_results, ignore_index=True)

    # Report filtering statistics (only once per experiment)
    pct_excluded = 100.0 * (1 - total_txns_after_filter / total_txns_before_filter) if total_txns_before_filter > 0 else 0
    active_window_ms = cooldown_start_ms - warmup_ms
    print(f" (warmup: {warmup_ms/1000:.0f}s, cooldown: {cooldown_ms/1000:.0f}s, " +
          f"active: {active_window_ms/1000:.0f}s, excluded {total_txns_before_filter - total_txns_after_filter}/{total_txns_before_filter} txns = {pct_excluded:.1f}%)", end='')

    return combined


def load_and_aggregate_results_consolidated(exp_info: Dict, reader: '_ConsolidatedReader') -> pd.DataFrame:
    """
    Load experiment results from consolidated parquet via a shared reader.

    Uses _ConsolidatedReader for efficient row-group-level access (file opened
    once, RG index reused across calls).  Warmup/cooldown filtering is applied
    in pandas after the read.

    Args:
        exp_info: Experiment information dict with 'label', 'hash', 'seeds', and 'config'
        reader: A _ConsolidatedReader instance for the consolidated parquet file

    Returns:
        DataFrame with aggregated statistics across all seeds (warmup/cooldown excluded)
    """
    # Compute warmup and cooldown durations for this experiment
    warmup_ms = compute_warmup_duration(exp_info['config'])
    cooldown_ms = compute_cooldown_duration(exp_info['config'])

    # Get simulation duration to compute cooldown threshold
    sim_duration_ms = exp_info['config'].get('simulation', {}).get('duration_ms', 3600000)
    cooldown_start_ms = sim_duration_ms - cooldown_ms

    df = reader.read_experiment(exp_info['label'], exp_info['hash'])
    if df is None or len(df) == 0:
        return None

    # Apply warmup/cooldown filter in pandas
    df = df[(df['t_submit'] >= warmup_ms) & (df['t_submit'] < cooldown_start_ms)]

    if len(df) == 0:
        return None

    # Report filtering statistics
    total_txns_after_filter = len(df)
    active_window_ms = cooldown_start_ms - warmup_ms

    # Estimate total txns (assuming uniform distribution over time)
    duration_ratio = sim_duration_ms / active_window_ms if active_window_ms > 0 else 1
    pct_excluded = 100.0 * (1 - 1 / duration_ratio) if duration_ratio > 1 else 0

    print(f" (warmup: {warmup_ms/1000:.0f}s, cooldown: {cooldown_ms/1000:.0f}s, " +
          f"active: {active_window_ms/1000:.0f}s, ~{total_txns_after_filter} active txns, est ~{pct_excluded:.1f}% excluded)", end='')

    return df


def compute_per_seed_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each seed separately.

    Args:
        df: DataFrame with 'seed' column and transaction data

    Returns:
        DataFrame with one row per seed containing statistics
    """
    if df is None or len(df) == 0 or 'seed' not in df.columns:
        return None

    seed_stats = []

    for seed in df['seed'].unique():
        seed_df = df[df['seed'] == seed]
        committed = seed_df[seed_df['status'] == 'committed']
        total = len(seed_df)

        if len(committed) == 0:
            continue

        # Calculate duration (in seconds)
        # After warmup/cooldown filtering, timestamps are still relative to simulation start
        # so we need max - min, not just max
        if len(seed_df) > 0:
            duration_s = (seed_df['t_submit'].max() - seed_df['t_submit'].min()) / 1000.0
        else:
            duration_s = 1.0

        # Compute overhead percentage
        committed_with_overhead = committed.copy()
        committed_with_overhead['overhead_pct'] = (
            100.0 * committed_with_overhead['commit_latency'] / committed_with_overhead['total_latency']
        )

        seed_stats.append({
            'seed': seed,
            'total_txns': total,
            'committed': len(committed),
            'success_rate': 100.0 * len(committed) / total,
            'p50_commit_latency': committed['commit_latency'].quantile(0.50),
            'p95_commit_latency': committed['commit_latency'].quantile(0.95),
            'p99_commit_latency': committed['commit_latency'].quantile(0.99),
            'throughput': len(committed) / duration_s,
            'mean_overhead_pct': committed_with_overhead['overhead_pct'].mean(),
            'p50_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.50),
            'p95_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.95),
            'p99_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.99)
        })

    return pd.DataFrame(seed_stats) if seed_stats else None


def compute_aggregate_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute aggregate statistics for an experiment.

    Returns statistics aggregated across all seeds, including mean and stddev.
    If no 'seed' column exists, computes statistics directly without stddev.
    """
    if df is None or len(df) == 0:
        return None

    committed = df[df['status'] == 'committed']
    total = len(df)

    if len(committed) == 0:
        return {
            'total_txns': total,
            'committed': 0,
            'success_rate': 0.0,
            'mean_commit_latency': None,
            'median_commit_latency': None,
            'p50_commit_latency': None,
            'p95_commit_latency': None,
            'p99_commit_latency': None,
            'mean_retries': None,
            'throughput': 0.0,
            'mean_overhead_pct': None,
            'p50_overhead_pct': None,
            'p95_overhead_pct': None,
            'p99_overhead_pct': None
        }

    # Try to compute per-seed statistics if 'seed' column exists
    per_seed_df = compute_per_seed_statistics(df)

    # If we have per-seed statistics, compute mean and stddev across seeds
    if per_seed_df is not None and len(per_seed_df) > 0:
        stats = {
            'num_seeds': len(per_seed_df),
            'total_txns': per_seed_df['total_txns'].sum(),
            'committed': per_seed_df['committed'].sum(),
            'aborted': per_seed_df['total_txns'].sum() - per_seed_df['committed'].sum(),
            'success_rate': per_seed_df['success_rate'].mean(),
            'success_rate_std': per_seed_df['success_rate'].std(),
            'p50_commit_latency': per_seed_df['p50_commit_latency'].mean(),
            'p50_commit_latency_std': per_seed_df['p50_commit_latency'].std(),
            'p95_commit_latency': per_seed_df['p95_commit_latency'].mean(),
            'p95_commit_latency_std': per_seed_df['p95_commit_latency'].std(),
            'p99_commit_latency': per_seed_df['p99_commit_latency'].mean(),
            'p99_commit_latency_std': per_seed_df['p99_commit_latency'].std(),
            'throughput': per_seed_df['throughput'].mean(),
            'throughput_std': per_seed_df['throughput'].std(),
            'mean_overhead_pct': per_seed_df['mean_overhead_pct'].mean(),
            'mean_overhead_pct_std': per_seed_df['mean_overhead_pct'].std(),
            'p50_overhead_pct': per_seed_df['p50_overhead_pct'].mean(),
            'p50_overhead_pct_std': per_seed_df['p50_overhead_pct'].std(),
            'p95_overhead_pct': per_seed_df['p95_overhead_pct'].mean(),
            'p95_overhead_pct_std': per_seed_df['p95_overhead_pct'].std(),
            'p99_overhead_pct': per_seed_df['p99_overhead_pct'].mean(),
            'p99_overhead_pct_std': per_seed_df['p99_overhead_pct'].std()
        }

        # Add some legacy statistics that aren't per-seed
        stats['mean_commit_latency'] = committed['commit_latency'].mean()
        stats['median_commit_latency'] = committed['commit_latency'].median()
        stats['mean_retries'] = committed['n_retries'].mean()
        stats['max_retries'] = committed['n_retries'].max()

        return stats

    # No seed column or single seed - compute statistics directly without stddev
    # This is for backward compatibility with tests and single-seed analysis
    # After warmup/cooldown filtering, timestamps are still relative to simulation start
    # so we need max - min, not just max
    if len(df) > 0:
        duration_s = (df['t_submit'].max() - df['t_submit'].min()) / 1000.0
    else:
        duration_s = 1.0

    # Compute overhead percentage
    committed_with_overhead = committed.copy()
    committed_with_overhead['overhead_pct'] = (
        100.0 * committed_with_overhead['commit_latency'] / committed_with_overhead['total_latency']
    )

    stats = {
        'total_txns': total,
        'committed': len(committed),
        'aborted': total - len(committed),
        'success_rate': 100.0 * len(committed) / total,
        'mean_commit_latency': committed['commit_latency'].mean(),
        'median_commit_latency': committed['commit_latency'].median(),
        'p50_commit_latency': committed['commit_latency'].quantile(0.50),
        'p95_commit_latency': committed['commit_latency'].quantile(0.95),
        'p99_commit_latency': committed['commit_latency'].quantile(0.99),
        'mean_retries': committed['n_retries'].mean(),
        'max_retries': committed['n_retries'].max(),
        'throughput': len(committed) / duration_s,
        'mean_overhead_pct': committed_with_overhead['overhead_pct'].mean(),
        'p50_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.50),
        'p95_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.95),
        'p99_overhead_pct': committed_with_overhead['overhead_pct'].quantile(0.99)
    }

    return stats


def build_experiment_index(base_dir: str, pattern: str) -> pd.DataFrame:
    """
    Build complete index of experiments with parameters and statistics.

    Returns:
        DataFrame with one row per experiment, including parameters and statistics.
    """
    experiments = scan_all_experiments(base_dir, pattern)

    if not experiments:
        raise ValueError(f"No experiments found in {base_dir} matching {pattern}")

    n_disk = sum(1 for e in experiments.values() if e['dir'] is not None)
    n_consolidated = len(experiments) - n_disk
    print(f"Found {len(experiments)} experiments ({n_disk} on disk, {n_consolidated} consolidated-only)")

    rows = []

    # Build consolidated reader once (reused across all experiments)
    use_consolidated = CONFIG.get('analysis', {}).get('use_consolidated', True)
    reader = None
    if use_consolidated:
        consolidated_filename = CONFIG.get('paths', {}).get('consolidated_file', 'experiments/consolidated.parquet')
        if os.path.isabs(consolidated_filename):
            consolidated_path = consolidated_filename
        elif consolidated_filename.startswith(base_dir):
            consolidated_path = consolidated_filename
        else:
            consolidated_path = os.path.join(base_dir, 'consolidated.parquet')
        reader = _ConsolidatedReader.open(consolidated_path)

    for exp_dir, exp_info in experiments.items():
        # Extract parameters
        params = extract_key_parameters(exp_info['config'])

        # Load and aggregate results
        print(f"Processing {exp_info['label']}-{exp_info['hash']}...", end='')

        # Use consolidated reader if available, otherwise use individual files
        if reader is not None:
            df = load_and_aggregate_results_consolidated(exp_info, reader)
            # Fall back to individual files if not in consolidated
            if df is None and exp_info['dir'] is not None:
                df = load_and_aggregate_results(exp_info)
        elif use_consolidated and exp_info['dir'] is not None:
            # Consolidated file doesn't exist, use individual files
            df = load_and_aggregate_results(exp_info)
        elif use_consolidated:
            print(" consolidated-only experiment but consolidated file missing")
            continue
        elif exp_info['dir'] is not None:
            df = load_and_aggregate_results(exp_info)
        else:
            print(" consolidated-only experiment but consolidated loading disabled")
            continue

        if df is None:
            print(" no data")
            continue

        stats = compute_aggregate_statistics(df)

        if stats is None:
            print(" no statistics")
            continue

        # Filter experiments with insufficient seeds
        num_seeds = len(exp_info['seeds'])
        min_seeds = CONFIG.get('analysis', {}).get('min_seeds', 3)
        if num_seeds < min_seeds:
            print(f" skipped (only {num_seeds} seeds, need {min_seeds})")
            continue

        print(f" {stats['committed']}/{stats['total_txns']} committed")

        # Combine into single row
        row = {
            'exp_dir': exp_dir,
            'label': exp_info['label'],
            'hash': exp_info['hash'],
            'num_seeds': num_seeds,
            **params,
            **stats
        }

        rows.append(row)

    index_df = pd.DataFrame(rows)

    return index_df


def plot_latency_vs_throughput(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Latency vs Throughput",
    group_by: str = None,
    annotate_success_rate: bool = False
):
    """
    Generate latency vs throughput plot (ANALYSIS_PLAN.md Figure 4.1).

    Args:
        index_df: Experiment index DataFrame
        output_path: Path to save plot
        title: Plot title
        group_by: Optional parameter to group by (e.g., 'num_tables', 'real_conflict_probability')
    """
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if group_by and group_by in index_df.columns:
        # Plot separate lines for each group
        groups = sorted(index_df[group_by].unique())

        for group_val in groups:
            subset = index_df[index_df[group_by] == group_val].copy()
            subset = subset.sort_values('throughput')

            # Plot P50, P95, P99
            for percentile, marker, alpha in [
                ('p50_commit_latency', 'o', 1.0),
                ('p95_commit_latency', 's', 0.7),
                ('p99_commit_latency', '^', 0.5)
            ]:
                label = f"{group_by}={group_val}, {percentile.replace('_commit_latency', '').upper()}"
                ax.plot(subset['throughput'], subset[percentile],
                       marker=marker, linewidth=2, markersize=8, alpha=alpha,
                       label=label)
    else:
        # Single series plot
        df_sorted = index_df.sort_values('throughput')

        # Check if stddev should be shown
        stddev_config = CONFIG.get('plots', {}).get('stddev', {})
        show_stddev = stddev_config.get('enabled', True)
        stddev_alpha = stddev_config.get('alpha', 0.2)

        # Plot P50, P95, P99 latency
        percentiles = [
            ('p50_commit_latency', 'P50', 'o', '#2E86AB'),
            ('p95_commit_latency', 'P95', 's', '#A23B72'),
            ('p99_commit_latency', 'P99', '^', '#F18F01')
        ]

        for col, label, marker, color in percentiles:
            ax.plot(df_sorted['throughput'], df_sorted[col],
                   marker=marker, linewidth=2.5, markersize=10,
                   label=label, color=color)

            # Add error bands if stddev is enabled and available
            if show_stddev and f'{col}_std' in df_sorted.columns:
                std_col = f'{col}_std'
                # Only plot if we have valid stddev values
                if df_sorted[std_col].notna().any():
                    ax.fill_between(
                        df_sorted['throughput'],
                        df_sorted[col] - df_sorted[std_col],
                        df_sorted[col] + df_sorted[std_col],
                        color=color,
                        alpha=stddev_alpha,
                        linewidth=0
                    )

    # Annotate success rate below P50 line for single-series plots
    if annotate_success_rate and not group_by and 'success_rate' in index_df.columns:
        df_sorted = index_df.sort_values('throughput')
        placed = []  # (x, y) of placed annotations for overlap detection
        for _, row in df_sorted.iterrows():
            sr = row.get('success_rate', 100.0)
            if sr < 99.5:
                x = row['throughput']
                y = row['p50_commit_latency']
                # Skip if too close to an existing annotation
                too_close = False
                for px, py in placed:
                    if abs(x - px) < 0.3 and abs(y - py) < 50:
                        too_close = True
                        break
                if too_close:
                    continue
                placed.append((x, y))
                ax.annotate(
                    f"{sr:.0f}%",
                    xy=(x, y),
                    xytext=(0, -14),
                    textcoords='offset points',
                    fontsize=8, color='#666666', ha='center', va='top'
                )

    # Mark saturation point (configurable success rate threshold)
    sat_config = CONFIG.get('plots', {}).get('saturation', {})
    sat_enabled = sat_config.get('enabled', False)  # Default: disabled
    sat_threshold = sat_config.get('threshold', 50.0)
    sat_tolerance = sat_config.get('tolerance', 5.0)

    if sat_enabled:
        saturation_points = index_df[index_df['success_rate'] < sat_threshold + sat_tolerance]
        if len(saturation_points) > 0:
            sat_throughput = saturation_points['throughput'].max()
            ax.axvline(sat_throughput, color='red', linestyle='--',
                      linewidth=2, alpha=0.5, label=f'~{sat_threshold:.0f}% Success Rate')
            ax.text(sat_throughput, ax.get_ylim()[1] * 0.9,
                   f'Saturation\n~{sat_throughput:.1f} commits/sec',
                   ha='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Achieved Throughput (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Commit Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"\nSaved latency vs throughput plot to {output_path}")
    plt.close()


def plot_sustainable_throughput(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Sustainable Throughput (99%+ Success Rate)",
    success_threshold: float = 99.0,
    group_by: str = None
):
    """
    Generate latency vs throughput plot showing only sustainable region.

    This plot filters to only show data points where success rate >= threshold,
    clearly demonstrating the throughput capacity each configuration can sustain
    without the confusing effects of saturation.

    Args:
        index_df: Experiment index DataFrame
        output_path: Path to save plot
        title: Plot title
        success_threshold: Minimum success rate to include (default 99%)
        group_by: Optional parameter to group by (e.g., 'label' for config comparison)
    """
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    # Filter to sustainable region only
    sustainable_df = index_df[index_df['success_rate'] >= success_threshold].copy()

    if sustainable_df.empty:
        print(f"  Skipping {output_path}: No data points with success rate >= {success_threshold}%")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Color palette for different configurations
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#44AA99', '#332288', '#DDCC77']

    if group_by and group_by in sustainable_df.columns:
        # Plot separate lines for each group (e.g., different optimization configs)
        groups = sorted(sustainable_df[group_by].unique())

        for i, group_val in enumerate(groups):
            subset = sustainable_df[sustainable_df[group_by] == group_val].copy()
            subset = subset.sort_values('throughput')

            if len(subset) == 0:
                continue

            color = colors[i % len(colors)]

            # Plot P50 latency as the main line
            ax.plot(subset['throughput'], subset['p50_commit_latency'],
                   marker='o', linewidth=2.5, markersize=10,
                   label=f'{group_val}', color=color)

            # Mark the maximum sustainable throughput with a vertical annotation
            max_throughput_row = subset.loc[subset['throughput'].idxmax()]
            max_tp = max_throughput_row['throughput']
            max_lat = max_throughput_row['p50_commit_latency']

            # Add a subtle marker at the ceiling
            ax.scatter([max_tp], [max_lat], s=200, marker='|', color=color,
                      linewidths=3, zorder=5)

    else:
        # Single series plot
        df_sorted = sustainable_df.sort_values('throughput')

        ax.plot(df_sorted['throughput'], df_sorted['p50_commit_latency'],
               marker='o', linewidth=2.5, markersize=10,
               label='P50 Latency', color='#2E86AB')

        # Show max sustainable throughput
        if len(df_sorted) > 0:
            max_tp = df_sorted['throughput'].max()
            ax.axvline(max_tp, color='green', linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Max Sustainable: {max_tp:.1f}/s')

    ax.set_xlabel('Achieved Throughput (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('P50 Commit Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n(Only showing success rate ≥ {success_threshold}%)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Add annotation explaining the plot
    ax.annotate(
        f'Each line ends at its maximum sustainable throughput\n'
        f'(highest throughput with ≥{success_threshold}% success)',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"\nSaved sustainable throughput plot to {output_path}")
    plt.close()


def plot_success_rate_vs_load(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Success Rate vs Offered Load"
):
    """Plot success rate and throughput vs offered load (inter-arrival time)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    df_sorted = index_df.sort_values('inter_arrival_scale')

    # Plot 1: Success rate vs inter-arrival time
    ax1.plot(df_sorted['inter_arrival_scale'], df_sorted['success_rate'],
            marker='o', linewidth=3, markersize=10, color='#2E86AB')

    # Add value labels
    for _, row in df_sorted.iterrows():
        if pd.notna(row['success_rate']):
            ax1.annotate(f"{row['success_rate']:.1f}%",
                       (row['inter_arrival_scale'], row['success_rate']),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=9)

    ax1.set_xlabel('Inter-arrival Time (ms)', fontsize=13)
    ax1.set_ylabel('Success Rate (%)', fontsize=13)
    ax1.set_title('Transaction Success Rate vs Load', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add saturation threshold line if enabled
    sat_config = CONFIG.get('plots', {}).get('saturation', {})
    sat_enabled = sat_config.get('enabled', False)  # Default: disabled
    sat_threshold = sat_config.get('threshold', 50.0)
    if sat_enabled:
        ax1.axhline(sat_threshold, color='red', linestyle='--', alpha=0.5,
                   label=f'{sat_threshold:.0f}% Saturation Threshold')

    ax1.legend()

    # Plot 2: Throughput vs inter-arrival time
    ax2.plot(df_sorted['inter_arrival_scale'], df_sorted['throughput'],
            marker='s', linewidth=3, markersize=10, color='#A23B72')

    # Add value labels
    for _, row in df_sorted.iterrows():
        if pd.notna(row['throughput']):
            ax2.annotate(f"{row['throughput']:.1f}",
                       (row['inter_arrival_scale'], row['throughput']),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=9)

    ax2.set_xlabel('Inter-arrival Time (ms)', fontsize=13)
    ax2.set_ylabel('Throughput (commits/sec)', fontsize=13)
    ax2.set_title('Achieved Throughput vs Load', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Saved success rate plot to {output_path}")
    plt.close()


def plot_success_rate_vs_throughput(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Transaction Success Rate vs Throughput",
    group_by: str = None
):
    """
    Plot success rate vs achieved throughput.

    Args:
        index_df: Experiment index DataFrame
        output_path: Path to save plot
        title: Plot title
        group_by: Optional parameter to group by (e.g., 'num_tables')
    """
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if group_by and group_by in index_df.columns:
        # Plot separate lines for each group
        groups = sorted(index_df[group_by].unique())

        for group_val in groups:
            subset = index_df[index_df[group_by] == group_val].copy()
            subset = subset.sort_values('throughput')

            ax.plot(subset['throughput'], subset['success_rate'],
                   marker='o', linewidth=2.5, markersize=8,
                   label=f"{group_by}={group_val}")
    else:
        # Single series plot
        df_sorted = index_df.sort_values('throughput')

        ax.plot(df_sorted['throughput'], df_sorted['success_rate'],
               marker='o', linewidth=3, markersize=10, color='#2E86AB',
               label='Success Rate')

        # Add error bands if stddev is enabled and available
        stddev_config = CONFIG.get('plots', {}).get('stddev', {})
        show_stddev = stddev_config.get('enabled', True)
        stddev_alpha = stddev_config.get('alpha', 0.2)

        if show_stddev and 'success_rate_std' in df_sorted.columns:
            if df_sorted['success_rate_std'].notna().any():
                ax.fill_between(
                    df_sorted['throughput'],
                    df_sorted['success_rate'] - df_sorted['success_rate_std'],
                    df_sorted['success_rate'] + df_sorted['success_rate_std'],
                    color='#2E86AB',
                    alpha=stddev_alpha,
                    linewidth=0
                )

    # Mark saturation threshold if enabled
    sat_config = CONFIG.get('plots', {}).get('saturation', {})
    sat_enabled = sat_config.get('enabled', False)  # Default: disabled
    sat_threshold = sat_config.get('threshold', 50.0)
    if sat_enabled:
        ax.axhline(sat_threshold, color='red', linestyle='--', linewidth=2, alpha=0.5,
                  label=f'{sat_threshold:.0f}% Saturation Threshold')

    ax.set_xlabel('Achieved Throughput (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.set_xlim(left=0)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Saved success rate vs throughput plot to {output_path}")
    plt.close()


def plot_overhead_vs_throughput(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Commit Overhead vs Throughput",
    group_by: str = None
):
    """
    Generate overhead percentage vs throughput plot.

    Overhead = (commit_latency / total_latency) * 100

    This represents the percentage of total transaction time spent in the commit
    protocol (retries, exponential backoff, manifest I/O operations) vs actual
    transaction execution.

    Args:
        index_df: Experiment index DataFrame
        output_path: Path to save plot
        title: Plot title
        group_by: Optional parameter to group by (e.g., 'num_tables')
    """
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    if group_by and group_by in index_df.columns:
        # Plot separate lines for each group
        groups = sorted(index_df[group_by].unique())

        for group_val in groups:
            subset = index_df[index_df[group_by] == group_val].copy()
            subset = subset.sort_values('throughput')

            # Plot P50, P95, P99 overhead
            for percentile, marker, alpha in [
                ('p50_overhead_pct', 'o', 1.0),
                ('p95_overhead_pct', 's', 0.7),
                ('p99_overhead_pct', '^', 0.5)
            ]:
                label = f"{group_by}={group_val}, {percentile.replace('_overhead_pct', '').upper()}"
                ax.plot(subset['throughput'], subset[percentile],
                       marker=marker, linewidth=2, markersize=8, alpha=alpha,
                       label=label)
    else:
        # Single series plot
        df_sorted = index_df.sort_values('throughput')

        # Plot P50, P95, P99 overhead
        percentiles = [
            ('p50_overhead_pct', 'P50', 'o', '#2E86AB'),
            ('p95_overhead_pct', 'P95', 's', '#A23B72'),
            ('p99_overhead_pct', 'P99', '^', '#F18F01')
        ]

        for col, label, marker, color in percentiles:
            ax.plot(df_sorted['throughput'], df_sorted[col],
                   marker=marker, linewidth=2.5, markersize=10,
                   label=label, color=color)

    ax.set_xlabel('Achieved Throughput (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Commit Overhead (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set reasonable axis limits
    ax.set_xlim(left=0)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Saved overhead vs throughput plot to {output_path}")
    plt.close()


class _ConsolidatedReader:
    """Efficient reader for consolidated parquet.

    Opens the file once, builds an ``(exp_name, exp_hash) → [rg_indices]``
    index from row-group metadata, and reads only matching row groups with
    column projection (excluding heavyweight metadata columns).
    """

    def __init__(self, pf):
        self.pf = pf
        self._index = self._build_index()
        # Exclude metadata columns that callers always drop
        exclude = {'config', 'code_hash', 'exp_name', 'exp_hash'}
        self._columns = [f.name for f in self.pf.schema_arrow if f.name not in exclude]

    @staticmethod
    def open(path: str) -> Optional['_ConsolidatedReader']:
        """Open a consolidated parquet file, returning None on failure."""
        if not os.path.exists(path):
            return None
        import pyarrow.parquet as pq
        try:
            pf = pq.ParquetFile(path)
        except Exception as e:
            print(f"Warning: Could not open consolidated file {path}: {e}")
            return None
        return _ConsolidatedReader(pf)

    def _build_index(self) -> Dict[Tuple[str, str], List[int]]:
        schema = self.pf.schema_arrow
        name_idx = schema.get_field_index('exp_name')
        hash_idx = schema.get_field_index('exp_hash')
        index: Dict[Tuple[str, str], List[int]] = {}
        for i in range(self.pf.metadata.num_row_groups):
            rg = self.pf.metadata.row_group(i)
            name_col = rg.column(name_idx)
            hash_col = rg.column(hash_idx)
            if not name_col.is_stats_set or not hash_col.is_stats_set:
                continue
            key = (name_col.statistics.min, hash_col.statistics.min)
            index.setdefault(key, []).append(i)
        return index

    def read_experiment(self, exp_name: str, exp_hash: str) -> Optional[pd.DataFrame]:
        """Read all rows for one experiment, returning a DataFrame or None."""
        rg_indices = self._index.get((exp_name, exp_hash))
        if not rg_indices:
            return None
        import pyarrow as pa
        tables = [self.pf.read_row_group(i, columns=self._columns) for i in rg_indices]
        return pa.concat_tables(tables).to_pandas()


def _load_from_consolidated(reader: '_ConsolidatedReader', exp_info: Dict) -> Optional[pd.DataFrame]:
    """Load raw transaction data for one experiment from consolidated parquet.

    Returns a DataFrame with the transaction columns (exp metadata columns
    dropped), or None if no rows match.
    """
    return reader.read_experiment(exp_info['label'], exp_info['hash'])


def plot_commit_rate_over_time(
    base_dir: str,
    pattern: str,
    output_path: str,
    title: str = "Commit Rate Over Time",
    window_size_sec: int = 60
):
    """
    Plot commit rate over time for all experiments matching pattern.

    Shows how commit throughput evolves during the simulation, useful for
    verifying steady-state and identifying transient behavior.

    Args:
        base_dir: Base directory containing experiments
        pattern: Pattern to match experiment directories
        output_path: Path to save plot
        title: Plot title
        window_size_sec: Window size in seconds for rate calculation
    """
    experiments = scan_all_experiments(base_dir, pattern)
    consolidated_path = os.path.join(base_dir, 'consolidated.parquet')
    reader = _ConsolidatedReader.open(consolidated_path)

    if not experiments:
        print(f"No experiments found for commit rate plot")
        return

    # Get warmup and cooldown durations for marking on plot
    first_exp = list(experiments.values())[0]
    first_config = first_exp['config']
    warmup_ms = compute_warmup_duration(first_config)
    cooldown_ms = compute_cooldown_duration(first_config)
    sim_duration_ms = first_config.get('simulation', {}).get('duration_ms', 3600000)
    warmup_min = warmup_ms / 60000
    cooldown_start_min = (sim_duration_ms - cooldown_ms) / 60000
    sim_duration_min = sim_duration_ms / 60000

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))

    for (exp_dir, exp_info), color in zip(experiments.items(), colors):
        # Load raw data (no warmup/cooldown filtering) for full time range
        all_results = []

        if exp_info['dir'] is None:
            # Consolidated-only: load from consolidated parquet
            if reader is not None:
                df = _load_from_consolidated(reader, exp_info)
                if df is not None and len(df) > 0:
                    all_results.append(df)
        else:
            for seed_dir in exp_info['seeds']:
                parquet_path = os.path.join(seed_dir, "results.parquet")
                if os.path.exists(parquet_path):
                    df = pd.read_parquet(parquet_path)
                    df['seed'] = os.path.basename(seed_dir)
                    all_results.append(df)

        if not all_results:
            continue

        df = pd.concat(all_results, ignore_index=True)

        if len(df) == 0:
            continue

        # Filter to committed transactions only
        committed = df[df['status'] == 'committed'].copy()

        if len(committed) == 0:
            continue

        # Compute commit rate in time windows
        window_size_ms = window_size_sec * 1000
        max_time_ms = committed['t_submit'].max()
        time_windows = np.arange(0, max_time_ms + window_size_ms, window_size_ms)

        rates = []
        window_midpoints = []

        for i in range(len(time_windows) - 1):
            start = time_windows[i]
            end = time_windows[i + 1]

            # Count commits in this window
            window_commits = committed[
                (committed['t_submit'] >= start) &
                (committed['t_submit'] < end)
            ]

            rate = len(window_commits) / window_size_sec  # commits/sec
            rates.append(rate)
            window_midpoints.append((start + end) / 2 / 60000)  # Convert to minutes

        # Extract key parameter for label
        config = exp_info['config']
        inter_arrival = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 'N/A')
        label = f"inter_arrival={inter_arrival}ms"

        ax.plot(window_midpoints, rates, linewidth=1.5, alpha=0.7,
               label=label, color=color)

    # Mark warmup period
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Warmup end ({warmup_min:.1f} min)')
    ax.axvspan(0, warmup_min, alpha=0.1, color='red')

    # Mark cooldown period
    ax.axvline(cooldown_start_min, color='blue', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Cooldown start ({cooldown_start_min:.1f} min)')
    ax.axvspan(cooldown_start_min, sim_duration_min, alpha=0.1, color='blue')

    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Commit Rate (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(0.5, -0.12), loc='upper center',
              fontsize=8, framealpha=0.9, ncol=4)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight')
    print(f"Saved commit rate over time plot to {output_path}")
    plt.close()


def format_value_with_stddev(value: float, stddev: float = None, decimals: int = 1) -> str:
    """
    Format a value with optional standard deviation.

    Args:
        value: The mean value to format
        stddev: Optional standard deviation (if None, only value is shown)
        decimals: Number of decimal places

    Returns:
        Formatted string like "12.5" or "12.5 ± 1.2"
    """
    if stddev is None or pd.isna(stddev):
        return f"{value:.{decimals}f}"
    else:
        return f"{value:.{decimals}f} ± {stddev:.{decimals}f}"


def generate_latency_vs_throughput_table(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = None,
    group_by: str = None
):
    """Generate markdown table for latency vs throughput data with optional stddev."""
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    # Check if stddev columns are available
    has_stddev = 'p50_commit_latency_std' in index_df.columns

    with open(output_path, 'w') as f:
        f.write("# Commit Latency vs Throughput\n\n")
        f.write("Analysis of how commit latency scales with achieved throughput.\n\n")
        if has_stddev:
            f.write("Values shown as mean ± standard deviation across seeds.\n\n")

        if group_by and group_by in index_df.columns:
            groups = sorted(index_df[group_by].unique())

            for group_val in groups:
                subset = index_df[index_df[group_by] == group_val].copy()
                subset = subset.sort_values('throughput')

                f.write(f"## {group_by} = {group_val}\n\n")
                f.write("| Throughput (c/s) | Success Rate (%) | P50 Latency (s) | P95 Latency (s) | P99 Latency (s) | Mean Retries |\n")
                f.write("|------------------|------------------|-----------------|-----------------|-----------------|---------------|\n")

                for _, row in subset.iterrows():
                    throughput = format_value_with_stddev(
                        row['throughput'],
                        row.get('throughput_std') if has_stddev else None,
                        decimals=1
                    )
                    success_rate = format_value_with_stddev(
                        row['success_rate'],
                        row.get('success_rate_std') if has_stddev else None,
                        decimals=1
                    )
                    p50 = format_value_with_stddev(
                        row['p50_commit_latency']/1000,
                        row.get('p50_commit_latency_std')/1000 if has_stddev and row.get('p50_commit_latency_std') else None,
                        decimals=2
                    )
                    p95 = format_value_with_stddev(
                        row['p95_commit_latency']/1000,
                        row.get('p95_commit_latency_std')/1000 if has_stddev and row.get('p95_commit_latency_std') else None,
                        decimals=2
                    )
                    p99 = format_value_with_stddev(
                        row['p99_commit_latency']/1000,
                        row.get('p99_commit_latency_std')/1000 if has_stddev and row.get('p99_commit_latency_std') else None,
                        decimals=2
                    )
                    f.write(f"| {throughput} | {success_rate} | {p50} | {p95} | {p99} | {row['mean_retries']:.1f} |\n")
                f.write("\n")
        else:
            df_sorted = index_df.sort_values('throughput')

            f.write("| Throughput (c/s) | Success Rate (%) | P50 Latency (s) | P95 Latency (s) | P99 Latency (s) | Mean Retries |\n")
            f.write("|------------------|------------------|-----------------|-----------------|-----------------|---------------|\n")

            for _, row in df_sorted.iterrows():
                throughput = format_value_with_stddev(
                    row['throughput'],
                    row.get('throughput_std') if has_stddev else None,
                    decimals=1
                )
                success_rate = format_value_with_stddev(
                    row['success_rate'],
                    row.get('success_rate_std') if has_stddev else None,
                    decimals=1
                )
                p50 = format_value_with_stddev(
                    row['p50_commit_latency']/1000,
                    row.get('p50_commit_latency_std')/1000 if has_stddev and row.get('p50_commit_latency_std') else None,
                    decimals=2
                )
                p95 = format_value_with_stddev(
                    row['p95_commit_latency']/1000,
                    row.get('p95_commit_latency_std')/1000 if has_stddev and row.get('p95_commit_latency_std') else None,
                    decimals=2
                )
                p99 = format_value_with_stddev(
                    row['p99_commit_latency']/1000,
                    row.get('p99_commit_latency_std')/1000 if has_stddev and row.get('p99_commit_latency_std') else None,
                    decimals=2
                )
                f.write(f"| {throughput} | {success_rate} | {p50} | {p95} | {p99} | {row['mean_retries']:.1f} |\n")

        f.write("\n## Notes\n\n")
        f.write("- Latencies reported in seconds (converted from milliseconds)\n")
        if has_stddev:
            f.write("- Values shown as mean ± standard deviation across multiple seeds\n")
        f.write("- Throughput = commits per second during steady-state window\n")
        f.write("- Success rate = percentage of transactions that committed successfully\n")
        f.write("- Mean retries = average number of retry attempts per committed transaction\n")

    print(f"Saved latency vs throughput table to {output_path}")


def generate_success_rate_table(
    index_df: pd.DataFrame,
    output_path: str,
    group_by: str = None
):
    """Generate markdown table for success rate data."""
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    with open(output_path, 'w') as f:
        f.write("# Transaction Success Rate vs Load\n\n")
        f.write("Analysis of how success rate changes with offered load (inter-arrival time).\n\n")

        if group_by and group_by in index_df.columns:
            groups = sorted(index_df[group_by].unique())

            for group_val in groups:
                subset = index_df[index_df[group_by] == group_val].copy()
                subset = subset.sort_values('inter_arrival_scale')

                f.write(f"## {group_by} = {group_val}\n\n")
                f.write("| Inter-Arrival (ms) | Arrival Rate (txn/s) | Throughput (c/s) | Success Rate (%) | Committed | Total |\n")
                f.write("|--------------------|----------------------|------------------|------------------|-----------|-------|\n")

                for _, row in subset.iterrows():
                    arrival_rate = 1000.0 / row['inter_arrival_scale']
                    f.write(f"| {row['inter_arrival_scale']:.0f} | {arrival_rate:.2f} | "
                           f"{row['throughput']:.1f} | {row['success_rate']:.1f} | "
                           f"{row['committed']} | {row['total_txns']} |\n")
                f.write("\n")
        else:
            df_sorted = index_df.sort_values('inter_arrival_scale')

            f.write("| Inter-Arrival (ms) | Arrival Rate (txn/s) | Throughput (c/s) | Success Rate (%) | Committed | Total |\n")
            f.write("|--------------------|----------------------|------------------|------------------|-----------|-------|\n")

            for _, row in df_sorted.iterrows():
                arrival_rate = 1000.0 / row['inter_arrival_scale']
                f.write(f"| {row['inter_arrival_scale']:.0f} | {arrival_rate:.2f} | "
                       f"{row['throughput']:.1f} | {row['success_rate']:.1f} | "
                       f"{row['committed']} | {row['total_txns']} |\n")

        f.write("\n## Notes\n\n")
        f.write("- Lower inter-arrival time = higher offered load\n")
        f.write("- Arrival rate = 1000 / inter_arrival_scale (transactions per second)\n")
        f.write("- Throughput may be less than arrival rate due to aborts\n")
        f.write("- Success rate typically drops as load increases due to contention\n")

    print(f"Saved success rate table to {output_path}")


def generate_overhead_table(
    index_df: pd.DataFrame,
    output_path: str,
    group_by: str = None
):
    """Generate markdown table for overhead percentage data with optional stddev."""
    # Handle empty or incomplete dataframes
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    # Check if stddev columns are available
    has_stddev = 'mean_overhead_pct_std' in index_df.columns

    with open(output_path, 'w') as f:
        f.write("# Commit Overhead vs Throughput\n\n")
        f.write("Analysis of commit protocol overhead as percentage of total transaction time.\n\n")
        f.write("**Overhead** = (commit_latency / total_latency) × 100\n\n")
        f.write("This represents time spent in retries, exponential backoff, and manifest I/O.\n\n")
        if has_stddev:
            f.write("Values shown as mean ± standard deviation across seeds.\n\n")

        if group_by and group_by in index_df.columns:
            groups = sorted(index_df[group_by].unique())

            for group_val in groups:
                subset = index_df[index_df[group_by] == group_val].copy()
                subset = subset.sort_values('throughput')

                f.write(f"## {group_by} = {group_val}\n\n")
                f.write("| Throughput (c/s) | Success Rate (%) | Mean Overhead (%) | P50 Overhead (%) | P95 Overhead (%) | P99 Overhead (%) |\n")
                f.write("|------------------|------------------|-------------------|------------------|------------------|------------------|\n")

                for _, row in subset.iterrows():
                    throughput = format_value_with_stddev(
                        row['throughput'],
                        row.get('throughput_std') if has_stddev else None,
                        decimals=1
                    )
                    success_rate = format_value_with_stddev(
                        row['success_rate'],
                        row.get('success_rate_std') if has_stddev else None,
                        decimals=1
                    )
                    mean_ovhd = format_value_with_stddev(
                        row['mean_overhead_pct'],
                        row.get('mean_overhead_pct_std') if has_stddev else None,
                        decimals=1
                    )
                    p50_ovhd = format_value_with_stddev(
                        row['p50_overhead_pct'],
                        row.get('p50_overhead_pct_std') if has_stddev else None,
                        decimals=1
                    )
                    p95_ovhd = format_value_with_stddev(
                        row['p95_overhead_pct'],
                        row.get('p95_overhead_pct_std') if has_stddev else None,
                        decimals=1
                    )
                    p99_ovhd = format_value_with_stddev(
                        row['p99_overhead_pct'],
                        row.get('p99_overhead_pct_std') if has_stddev else None,
                        decimals=1
                    )
                    f.write(f"| {throughput} | {success_rate} | {mean_ovhd} | {p50_ovhd} | {p95_ovhd} | {p99_ovhd} |\n")
                f.write("\n")
        else:
            df_sorted = index_df.sort_values('throughput')

            f.write("| Throughput (c/s) | Success Rate (%) | Mean Overhead (%) | P50 Overhead (%) | P95 Overhead (%) | P99 Overhead (%) |\n")
            f.write("|------------------|------------------|-------------------|------------------|------------------|------------------|\n")

            for _, row in df_sorted.iterrows():
                throughput = format_value_with_stddev(
                    row['throughput'],
                    row.get('throughput_std') if has_stddev else None,
                    decimals=1
                )
                success_rate = format_value_with_stddev(
                    row['success_rate'],
                    row.get('success_rate_std') if has_stddev else None,
                    decimals=1
                )
                mean_ovhd = format_value_with_stddev(
                    row['mean_overhead_pct'],
                    row.get('mean_overhead_pct_std') if has_stddev else None,
                    decimals=1
                )
                p50_ovhd = format_value_with_stddev(
                    row['p50_overhead_pct'],
                    row.get('p50_overhead_pct_std') if has_stddev else None,
                    decimals=1
                )
                p95_ovhd = format_value_with_stddev(
                    row['p95_overhead_pct'],
                    row.get('p95_overhead_pct_std') if has_stddev else None,
                    decimals=1
                )
                p99_ovhd = format_value_with_stddev(
                    row['p99_overhead_pct'],
                    row.get('p99_overhead_pct_std') if has_stddev else None,
                    decimals=1
                )
                f.write(f"| {throughput} | {success_rate} | {mean_ovhd} | {p50_ovhd} | {p95_ovhd} | {p99_ovhd} |\n")

        f.write("\n## Interpretation\n\n")
        f.write("- **Low overhead (<10%)**: System operating efficiently, minimal contention\n")
        f.write("- **Medium overhead (10-30%)**: Moderate contention, acceptable performance\n")
        f.write("- **High overhead (30-50%)**: High contention, commit protocol becoming significant\n")
        f.write("- **Very high overhead (>50%)**: Commit protocol dominates, system saturated\n\n")
        if has_stddev:
            f.write("Values shown as mean ± standard deviation across multiple seeds.\n")
        f.write("At saturation, overhead can exceed 50%, meaning more time is spent retrying\n")
        f.write("commits than executing transactions!\n")

    print(f"Saved overhead table to {output_path}")


def generate_success_rate_vs_throughput_table(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = None,
    group_by: str = None
):
    """Generate markdown table for success rate vs throughput data with optional stddev."""
    if index_df.empty or 'throughput' not in index_df.columns:
        print(f"  Skipping {output_path}: No data or missing required columns")
        return

    has_stddev = 'success_rate_std' in index_df.columns

    with open(output_path, 'w') as f:
        f.write("# Success Rate vs Throughput\n\n")
        f.write("Analysis of how transaction success rate degrades with achieved throughput.\n\n")
        if has_stddev:
            f.write("Values shown as mean ± standard deviation across seeds.\n\n")

        if group_by and group_by in index_df.columns:
            groups = sorted(index_df[group_by].unique())

            for group_val in groups:
                subset = index_df[index_df[group_by] == group_val].copy()
                subset = subset.sort_values('throughput')

                f.write(f"## {group_by} = {group_val}\n\n")
                f.write("| Throughput (c/s) | Success Rate (%) |\n")
                f.write("|------------------|------------------|\n")

                for _, row in subset.iterrows():
                    throughput = format_value_with_stddev(
                        row['throughput'],
                        row.get('throughput_std') if has_stddev else None,
                        decimals=1
                    )
                    success_rate = format_value_with_stddev(
                        row['success_rate'],
                        row.get('success_rate_std') if has_stddev else None,
                        decimals=1
                    )
                    f.write(f"| {throughput} | {success_rate} |\n")
                f.write("\n")
        else:
            df_sorted = index_df.sort_values('throughput')

            f.write("| Throughput (c/s) | Success Rate (%) |\n")
            f.write("|------------------|------------------|\n")

            for _, row in df_sorted.iterrows():
                throughput = format_value_with_stddev(
                    row['throughput'],
                    row.get('throughput_std') if has_stddev else None,
                    decimals=1
                )
                success_rate = format_value_with_stddev(
                    row['success_rate'],
                    row.get('success_rate_std') if has_stddev else None,
                    decimals=1
                )
                f.write(f"| {throughput} | {success_rate} |\n")

        f.write("\n## Notes\n\n")
        if has_stddev:
            f.write("- Values shown as mean ± standard deviation across multiple seeds\n")
        f.write("- Throughput = commits per second during steady-state window\n")
        f.write("- Success rate = percentage of transactions that committed successfully\n")
        f.write("- Success rate typically degrades as throughput increases due to contention\n")

    print(f"Saved success rate vs throughput table to {output_path}")


def generate_commit_rate_over_time_table(
    base_dir: str,
    pattern: str,
    output_path: str,
    title: str = None,
    window_size_sec: int = 60
):
    """Generate markdown table for commit rate over time data."""
    experiments = scan_all_experiments(base_dir, pattern)
    consolidated_path = os.path.join(base_dir, 'consolidated.parquet')
    reader = _ConsolidatedReader.open(consolidated_path)

    if not experiments:
        print(f"  Skipping {output_path}: No experiments found")
        return

    with open(output_path, 'w') as f:
        f.write("# Commit Rate Over Time\n\n")
        f.write(f"Commit throughput in {window_size_sec}-second windows.\n\n")

        for exp_dir, exp_info in experiments.items():
            all_results = []

            if exp_info['dir'] is None:
                if reader is not None:
                    df = _load_from_consolidated(reader, exp_info)
                    if df is not None and len(df) > 0:
                        all_results.append(df)
            else:
                for seed_dir in exp_info['seeds']:
                    parquet_path = os.path.join(seed_dir, "results.parquet")
                    if os.path.exists(parquet_path):
                        df = pd.read_parquet(parquet_path)
                        df['seed'] = os.path.basename(seed_dir)
                        all_results.append(df)

            if not all_results:
                continue

            df = pd.concat(all_results, ignore_index=True)
            if len(df) == 0:
                continue

            committed = df[df['status'] == 'committed'].copy()
            if len(committed) == 0:
                continue

            # Compute commit rate in time windows
            window_size_ms = window_size_sec * 1000
            max_time_ms = committed['t_submit'].max()
            time_windows = np.arange(0, max_time_ms + window_size_ms, window_size_ms)

            config = exp_info['config']
            inter_arrival = config.get('transaction', {}).get('inter_arrival', {}).get('scale', 'N/A')
            f.write(f"## inter_arrival={inter_arrival}ms\n\n")
            f.write("| Time (min) | Commit Rate (c/s) |\n")
            f.write("|------------|-------------------|\n")

            for i in range(len(time_windows) - 1):
                start = time_windows[i]
                end = time_windows[i + 1]
                window_commits = committed[
                    (committed['t_submit'] >= start) &
                    (committed['t_submit'] < end)
                ]
                rate = len(window_commits) / window_size_sec
                midpoint_min = (start + end) / 2 / 60000
                f.write(f"| {midpoint_min:.1f} | {rate:.1f} |\n")

            f.write("\n")

        f.write("## Notes\n\n")
        f.write(f"- Rates computed in {window_size_sec}-second windows\n")
        f.write("- Time shown as minutes from simulation start\n")
        f.write("- Includes warmup and cooldown periods\n")

    print(f"Saved commit rate over time table to {output_path}")


def save_experiment_index(index_df: pd.DataFrame, output_path: str):
    """Save experiment index to CSV for reference."""
    # Handle empty dataframes
    if index_df.empty:
        # Create empty DataFrame with expected column headers
        columns = [
            'label', 'hash', 'num_seeds',
            'inter_arrival_scale', 'num_tables',
            'real_conflict_probability',
            'total_txns', 'committed', 'success_rate',
            'throughput',
            'mean_commit_latency', 'p50_commit_latency',
            'p95_commit_latency', 'p99_commit_latency',
            'mean_retries',
            'mean_overhead_pct', 'p50_overhead_pct',
            'p95_overhead_pct', 'p99_overhead_pct'
        ]
        df_export = pd.DataFrame(columns=columns)
        df_export.to_csv(output_path, index=False)
        print(f"Saved empty experiment index to {output_path}")
        return

    # Select relevant columns
    columns = [
        'label', 'hash', 'num_seeds',
        'inter_arrival_scale', 'num_tables',
        'real_conflict_probability',
        'partition_enabled', 'num_partitions',
        'partitions_per_txn_mean', 'partitions_per_txn_max',
        'total_txns', 'committed', 'success_rate',
        'throughput',
        'mean_commit_latency', 'p50_commit_latency',
        'p95_commit_latency', 'p99_commit_latency',
        'mean_retries',
        'mean_overhead_pct', 'p50_overhead_pct',
        'p95_overhead_pct', 'p99_overhead_pct'
    ]

    # Filter to only columns that exist
    available_cols = [col for col in columns if col in index_df.columns]

    df_export = index_df[available_cols].copy()
    df_export.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved experiment index to {output_path}")


def parse_filter_expression(filter_str: str) -> Tuple[str, str, any]:
    """
    Parse a filter expression like "t_cas_mean==50" or "num_tables>=5".

    Supported operators: ==, !=, <, <=, >, >=

    Args:
        filter_str: Filter expression string

    Returns:
        Tuple of (parameter_name, operator, value)

    Raises:
        ValueError: If filter expression is invalid

    Examples:
        >>> parse_filter_expression("t_cas_mean==50")
        ('t_cas_mean', '==', 50)
        >>> parse_filter_expression("num_tables>=5")
        ('num_tables', '>=', 5)
    """
    # Strip whitespace
    filter_str = filter_str.strip()

    # Try to match operators (order matters - check >= before >)
    operators = ['==', '!=', '<=', '>=', '<', '>']

    for op in operators:
        if op in filter_str:
            parts = filter_str.split(op, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid filter expression: {filter_str}")

            param_name = parts[0].strip()
            value_str = parts[1].strip()

            # Validate that parameter name and value are not empty
            if not param_name or not value_str:
                raise ValueError(f"Invalid filter expression: {filter_str}")

            # Try to convert value to appropriate type
            try:
                # Try int first
                value = int(value_str)
            except ValueError:
                try:
                    # Try float
                    value = float(value_str)
                except ValueError:
                    # Keep as string (strip quotes if present)
                    value = value_str.strip('"').strip("'")

            return (param_name, op, value)

    raise ValueError(f"No valid operator found in filter expression: {filter_str}")


def apply_filters(df: pd.DataFrame, filter_expressions: List[str]) -> pd.DataFrame:
    """
    Apply parameter filters to experiment index DataFrame.

    Args:
        df: Experiment index DataFrame
        filter_expressions: List of filter expressions (e.g., ["t_cas_mean==50", "num_tables>=5"])

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If filter references non-existent column

    Examples:
        >>> df = pd.DataFrame({'t_cas_mean': [15, 50, 100], 'num_tables': [1, 5, 10]})
        >>> filtered = apply_filters(df, ["t_cas_mean==50"])
        >>> len(filtered)
        1
        >>> filtered = apply_filters(df, ["t_cas_mean>=50", "num_tables<=5"])
        >>> len(filtered)
        1
    """
    if not filter_expressions:
        return df

    result = df.copy()

    for filter_expr in filter_expressions:
        param_name, op, value = parse_filter_expression(filter_expr)

        # Check if column exists
        if param_name not in result.columns:
            available = [c for c in result.columns if not c.startswith('_')]
            raise ValueError(
                f"Filter parameter '{param_name}' not found in DataFrame. "
                f"Available parameters: {', '.join(available)}"
            )

        # Apply filter based on operator
        if op == '==':
            mask = result[param_name] == value
        elif op == '!=':
            mask = result[param_name] != value
        elif op == '<':
            mask = result[param_name] < value
        elif op == '<=':
            mask = result[param_name] <= value
        elif op == '>':
            mask = result[param_name] > value
        elif op == '>=':
            mask = result[param_name] >= value
        else:
            raise ValueError(f"Unsupported operator: {op}")

        result = result[mask]
        print(f"Applied filter '{filter_expr}': {len(result)} experiments remain")

    return result


# ---------------------------------------------------------------------------
# Heatmap generation (moved from scripts/plot_heatmap.py)
# ---------------------------------------------------------------------------

def _extract_heatmap_params(experiments_dir: str, pattern: str,
                            x_param: str, y_param: str,
                            extra_params: list = None,
                            experiments_cache: dict = None) -> pd.DataFrame:
    """Extract parameters for heatmap from experiment directories and consolidated.

    Args:
        extra_params: Additional parameter names to extract (e.g. for filtering).
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
                          If provided, skips redundant directory scan.
    """
    extra_params = extra_params or []
    records = []

    experiments = experiments_cache if experiments_cache is not None else scan_all_experiments(experiments_dir, pattern)

    for dir_key, exp_info in experiments.items():
        cfg = exp_info['config']

        x_val = _extract_param_value(cfg, x_param)
        y_val = _extract_param_value(cfg, y_param)
        if x_val is None or y_val is None:
            continue

        record = {
            "exp_dir": exp_info['dir'],
            x_param: x_val,
            y_param: y_val,
            "num_seeds": len(exp_info['seeds']),
        }
        if exp_info['dir'] is None:
            record["_exp_label"] = exp_info['label']
            record["_exp_hash"] = exp_info['hash']
        for ep in extra_params:
            record[ep] = _extract_param_value(cfg, ep)
        records.append(record)

    return pd.DataFrame(records)


def _extract_param_value(cfg: dict, param_name: str):
    """Extract a parameter value from a config dict.

    Supports flattened names like 'inter_arrival_scale' and 'fast_append_ratio'.
    """
    txn = cfg.get("transaction", {})
    catalog = cfg.get("catalog", {})

    # Map commonly used flattened names to config paths
    param_map = {
        "inter_arrival_scale": lambda: txn.get("inter_arrival", {}).get("scale"),
        "fast_append_ratio": lambda: txn.get("operation_types", {}).get("fast_append"),
        "real_conflict_probability": lambda: txn.get("real_conflict_probability"),
        "num_tables": lambda: catalog.get("num_tables"),
        "catalog_service_latency_ms": lambda: catalog.get("service", {}).get("latency_ms"),
        "storage_provider": lambda: cfg.get("storage", {}).get("provider"),
    }

    if param_name in param_map:
        return param_map[param_name]()

    # Fallback: try direct lookup in common sections
    for section in [txn, catalog, cfg.get("storage", {}), cfg.get("simulation", {})]:
        if param_name in section:
            return section[param_name]

    return None


def _compute_seed_metrics(df: pd.DataFrame, op_type_filter: str = None) -> Optional[Dict]:
    """Compute per-seed heatmap metrics from a transaction DataFrame."""
    if op_type_filter:
        if "operation_type" not in df.columns:
            return None
        df = df[df["operation_type"] == op_type_filter]
        if len(df) == 0:
            return None

    total = len(df)
    committed = (df["status"].str.lower() == "committed").sum()

    # Filter to steady state (middle 80%)
    duration = df["t_commit"].max() - df["t_submit"].min()
    warmup = duration * 0.1
    t_min = df["t_submit"].min() + warmup
    t_max = df["t_commit"].max() - duration * 0.1
    df_steady = df[(df["t_submit"] >= t_min) & (df["t_commit"] <= t_max)]

    steady_committed = df_steady[df_steady["status"].str.lower() == "committed"]
    if len(steady_committed) == 0:
        return None

    throughput = len(steady_committed) / (duration / 1000 / 3600)
    latency = steady_committed["commit_latency"]

    return {
        "success_rate": committed / total * 100 if total > 0 else 0,
        "throughput": throughput,
        "mean_latency": latency.mean(),
        "p50_latency": latency.median(),
        "p95_latency": latency.quantile(0.95),
        "p99_latency": latency.quantile(0.99),
    }


def _load_heatmap_results(params_df: pd.DataFrame,
                          x_param: str, y_param: str,
                          op_type_filter: str = None) -> pd.DataFrame:
    """Load and aggregate results for heatmap generation.

    Args:
        params_df: DataFrame with experiment directories and parameters.
        x_param: Parameter name for X axis.
        y_param: Parameter name for Y axis.
        op_type_filter: If set, filter to this operation_type before computing metrics.
                        E.g. 'fast_append' or 'validated_overwrite'.
    """
    all_results = []

    # Determine consolidated path for consolidated-only experiments
    consolidated_path = None
    if "_exp_label" in params_df.columns:
        # There may be consolidated-only rows — find the consolidated file
        # Use the base_dir from a directory-based row, or infer from CONFIG
        for _, r in params_df.iterrows():
            if r["exp_dir"] is not None:
                consolidated_path = os.path.join(
                    str(Path(r["exp_dir"]).parent), 'consolidated.parquet')
                break
        if consolidated_path is None:
            consolidated_path = CONFIG.get('paths', {}).get(
                'consolidated_file', 'experiments/consolidated.parquet')

    reader = _ConsolidatedReader.open(consolidated_path) if consolidated_path else None

    for _, row in params_df.iterrows():
        seed_results = []

        if row["exp_dir"] is None:
            # Consolidated-only experiment
            if reader is not None:
                exp_info = {'label': row["_exp_label"], 'hash': row["_exp_hash"]}
                df = _load_from_consolidated(reader, exp_info)
                if df is not None and len(df) > 0:
                    # Split by seed and compute per-seed metrics
                    for seed_val, seed_df in df.groupby('seed'):
                        metrics = _compute_seed_metrics(seed_df, op_type_filter)
                        if metrics:
                            seed_results.append(metrics)
        else:
            exp_dir = Path(row["exp_dir"])
            for seed_dir in exp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                results_path = seed_dir / "results.parquet"
                if not results_path.exists():
                    continue

                df = pd.read_parquet(results_path)
                metrics = _compute_seed_metrics(df, op_type_filter)
                if metrics:
                    seed_results.append(metrics)

        if seed_results:
            seed_df = pd.DataFrame(seed_results)
            all_results.append({
                x_param: row[x_param],
                y_param: row[y_param],
                "num_seeds": len(seed_results),
                "success_rate": seed_df["success_rate"].mean(),
                "throughput": seed_df["throughput"].mean(),
                "mean_latency": seed_df["mean_latency"].mean(),
                "p50_latency": seed_df["p50_latency"].mean(),
                "p95_latency": seed_df["p95_latency"].mean(),
                "p99_latency": seed_df["p99_latency"].mean(),
            })

    return pd.DataFrame(all_results)


def _load_raw_heatmap_data(params_df: pd.DataFrame) -> dict:
    """Load raw seed DataFrames for all experiments in params_df.

    Returns dict keyed by experiment identifier, containing list of
    per-seed DataFrames.  This allows computing metrics for multiple
    op_type filters without re-reading parquet files from disk.
    """
    raw_data = {}

    # Set up consolidated reader if needed
    consolidated_path = None
    if "_exp_label" in params_df.columns:
        for _, r in params_df.iterrows():
            if r["exp_dir"] is not None:
                consolidated_path = os.path.join(
                    str(Path(r["exp_dir"]).parent), 'consolidated.parquet')
                break
        if consolidated_path is None:
            consolidated_path = CONFIG.get('paths', {}).get(
                'consolidated_file', 'experiments/consolidated.parquet')

    reader = _ConsolidatedReader.open(consolidated_path) if consolidated_path else None

    for idx, row in params_df.iterrows():
        if row["exp_dir"] is not None:
            exp_key = str(row["exp_dir"])
        elif "_exp_label" in params_df.columns:
            exp_key = f"{row['_exp_label']}-{row['_exp_hash']}"
        else:
            continue

        seed_dfs = []

        if row["exp_dir"] is None:
            if reader is not None:
                exp_info = {'label': row["_exp_label"], 'hash': row["_exp_hash"]}
                df = _load_from_consolidated(reader, exp_info)
                if df is not None and len(df) > 0:
                    for seed_val, seed_df in df.groupby('seed'):
                        seed_dfs.append(seed_df)
        else:
            exp_dir = Path(row["exp_dir"])
            for seed_dir in exp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                results_path = seed_dir / "results.parquet"
                if not results_path.exists():
                    continue
                seed_dfs.append(pd.read_parquet(results_path))

        if seed_dfs:
            raw_data[exp_key] = seed_dfs

    return raw_data


def _aggregate_heatmap_metrics(raw_data: dict, params_df: pd.DataFrame,
                                x_param: str, y_param: str,
                                op_type_filter: str = None) -> pd.DataFrame:
    """Compute aggregated metrics from cached raw data.

    Args:
        raw_data: Dict keyed by experiment identifier, containing per-seed DataFrames.
        params_df: DataFrame with experiment parameters.
        x_param, y_param: Parameter names for axes.
        op_type_filter: Optional operation type to filter by before computing metrics.
    """
    all_results = []

    for _, row in params_df.iterrows():
        if row["exp_dir"] is not None:
            exp_key = str(row["exp_dir"])
        elif "_exp_label" in params_df.columns:
            exp_key = f"{row['_exp_label']}-{row['_exp_hash']}"
        else:
            continue

        seed_dfs = raw_data.get(exp_key, [])
        seed_results = []
        for seed_df in seed_dfs:
            metrics = _compute_seed_metrics(seed_df, op_type_filter)
            if metrics:
                seed_results.append(metrics)

        if seed_results:
            agg_df = pd.DataFrame(seed_results)
            all_results.append({
                x_param: row[x_param],
                y_param: row[y_param],
                "num_seeds": len(seed_results),
                "success_rate": agg_df["success_rate"].mean(),
                "throughput": agg_df["throughput"].mean(),
                "mean_latency": agg_df["mean_latency"].mean(),
                "p50_latency": agg_df["p50_latency"].mean(),
                "p95_latency": agg_df["p95_latency"].mean(),
                "p99_latency": agg_df["p99_latency"].mean(),
            })

    return pd.DataFrame(all_results)


def _create_single_heatmap(data: pd.DataFrame, x_param: str, y_param: str,
                           value_col: str, title: str, output_path,
                           cmap: str = "viridis", vmin=None, vmax=None,
                           fmt: str = ".1f", figsize=None, dpi: int = 100,
                           success_rate_data: pd.DataFrame = None,
                           implausible_threshold: float = 95.0,
                           unreliable_threshold: float = 80.0):
    """Create a single heatmap from 2D parameter sweep data."""
    pivot = data.pivot_table(
        index=y_param, columns=x_param,
        values=value_col, aggfunc="mean"
    )
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=figsize or (12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" if isinstance(c, float) and c == int(c) else f"{c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" if isinstance(r, float) else f"{r}" for r in pivot.index])

    ax.set_xlabel(x_param.replace("_", " ").title())
    ax.set_ylabel(y_param.replace("_", " ").title())
    ax.set_title(title, fontsize=14, fontweight="bold")

    fig.colorbar(im, ax=ax)

    # Use actual colormap luminance for text color (works with all colormaps)
    norm = plt.Normalize(vmin=vmin or np.nanmin(pivot.values),
                         vmax=vmax or np.nanmax(pivot.values))
    colormap = plt.get_cmap(cmap)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                rgba = colormap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        color=text_color, fontsize=8)

    if success_rate_data is not None and value_col != "success_rate":
        sr_pivot = success_rate_data.pivot_table(
            index=y_param, columns=x_param,
            values="success_rate", aggfunc="mean"
        )
        sr_pivot = sr_pivot.reindex(index=pivot.index, columns=pivot.columns)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                sr = sr_pivot.values[i, j]
                if np.isnan(sr):
                    continue
                val = pivot.values[i, j]
                if not np.isnan(val):
                    rgba = colormap(norm(val))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    hatch_color = 'white' if lum < 0.5 else 'black'
                else:
                    hatch_color = 'black'
                if sr < unreliable_threshold:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch='xx', edgecolor=hatch_color,
                        linewidth=0, alpha=0.5)
                    ax.add_patch(rect)
                elif sr < implausible_threshold:
                    rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                        fill=False, hatch='//', edgecolor=hatch_color,
                        linewidth=0, alpha=0.5)
                    ax.add_patch(rect)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='none', edgecolor='gray', hatch='//',
                  label=f'Success < {implausible_threshold:.0f}%'),
            Patch(facecolor='none', edgecolor='gray', hatch='xx',
                  label=f'Success < {unreliable_threshold:.0f}%'),
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                  fontsize=7, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Saved: {output_path}")


def generate_heatmap_plots(base_dir: str, pattern: str, output_dir: str,
                           x_param: str, y_param: str, metrics: list,
                           config: dict = None,
                           experiments_cache: dict = None):
    """Generate heatmap plots for a 2D parameter sweep.

    Args:
        base_dir: Directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        output_dir: Directory to write plots.
        x_param: Parameter name for X axis.
        y_param: Parameter name for Y axis.
        metrics: List of metric names to generate heatmaps for.
        config: Optional plotting config overrides.
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
                          If provided, avoids redundant directory scanning.
    """
    config = config or {}
    os.makedirs(output_dir, exist_ok=True)
    figsize = config.get("figsize", [12, 8])
    cmap = config.get("cmap", "viridis")
    fmt = config.get("fmt", ".1f")
    dpi = config.get("dpi", DEFAULT_DPI)
    filters = config.get("filters", [])
    title_suffix = config.get("title_suffix", "")
    implausible_threshold = config.get("implausible_threshold", 95.0)
    unreliable_threshold = config.get("unreliable_threshold", 80.0)
    xheatmap_only = config.get("xheatmap_only", False)

    # Determine extra params needed for filtering
    extra_params = []
    for f in filters:
        param_name, _, _ = parse_filter_expression(f)
        if param_name not in (x_param, y_param) and param_name not in extra_params:
            extra_params.append(param_name)

    print(f"Generating heatmaps for {pattern}...")
    params_df = _extract_heatmap_params(base_dir, pattern, x_param, y_param,
                                        extra_params=extra_params,
                                        experiments_cache=experiments_cache)
    if len(params_df) == 0:
        print(f"  No experiments found matching {pattern}")
        return

    if filters:
        params_df = apply_filters(params_df, filters)
        if len(params_df) == 0:
            print(f"  No experiments remain after filtering")
            return
        # Drop extra filter columns before passing to metric computation
        drop_cols = [c for c in extra_params if c in params_df.columns]
        if drop_cols:
            params_df = params_df.drop(columns=drop_cols)

    # Load raw seed data once; compute metrics for aggregate and per-type
    # from the cached data to avoid redundant parquet reads.
    per_type_metrics = config.get("per_type_metrics", [])
    need_raw_cache = bool(per_type_metrics)

    if need_raw_cache:
        raw_data = _load_raw_heatmap_data(params_df)
        results_df = _aggregate_heatmap_metrics(raw_data, params_df, x_param, y_param)
    else:
        raw_data = None
        results_df = _load_heatmap_results(params_df, x_param, y_param)

    if len(results_df) == 0:
        print(f"  No results loaded for {pattern}")
        return

    # Save combined data
    results_df.to_csv(os.path.join(output_dir, "heatmap_data.csv"), index=False)

    metric_configs = {
        "success_rate": ("Success Rate (%)", "RdYlBu", 60, 100),
        "throughput": ("Throughput (commits/hour)", "plasma", None, None),
        "mean_latency": ("Mean Commit Latency (ms)", "inferno_r", None, None),
        "p99_latency": ("P99 Commit Latency (ms)", "inferno_r", None, None),
        "p95_latency": ("P95 Commit Latency (ms)", "inferno_r", None, None),
        "p50_latency": ("P50 Commit Latency (ms)", "inferno_r", None, None),
    }

    for metric in metrics:
        if metric not in results_df.columns:
            print(f"  Skipping metric '{metric}' (not in results)")
            continue

        metric_label, m_cmap, m_vmin, m_vmax = metric_configs.get(
            metric, (metric.replace("_", " ").title(), cmap, None, None)
        )
        m_fmt = ".0f" if metric in ("throughput", "mean_latency", "p99_latency", "p95_latency", "p50_latency") else fmt

        if not xheatmap_only or metric == "success_rate":
            _create_single_heatmap(
                results_df, x_param, y_param, metric,
                title=f"{metric_label} by {y_param.replace('_', ' ').title()} and {x_param.replace('_', ' ').title()}{title_suffix}",
                output_path=os.path.join(output_dir, f"heatmap_{metric}.png"),
                cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, fmt=m_fmt,
                figsize=figsize, dpi=dpi,
            )

        if metric != "success_rate":
            _create_single_heatmap(
                results_df, x_param, y_param, metric,
                title=f"{metric_label} by {y_param.replace('_', ' ').title()} and {x_param.replace('_', ' ').title()}{title_suffix}",
                output_path=os.path.join(output_dir, f"xheatmap_{metric}.png"),
                cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, fmt=m_fmt,
                figsize=figsize, dpi=dpi,
                success_rate_data=results_df,
                implausible_threshold=implausible_threshold,
                unreliable_threshold=unreliable_threshold,
            )

    # Per-operation-type heatmaps — reuse cached raw data
    if per_type_metrics:
        for op_type in ["fast_append", "validated_overwrite"]:
            prefix = "fa" if op_type == "fast_append" else "vo"
            print(f"  Computing per-type metrics for {op_type}...")
            type_results = _aggregate_heatmap_metrics(
                raw_data, params_df, x_param, y_param, op_type_filter=op_type)
            if len(type_results) == 0:
                print(f"    No data for {op_type}")
                continue

            op_label = "FastAppend" if op_type == "fast_append" else "ValidatedOverwrite"
            for metric in per_type_metrics:
                if metric not in type_results.columns:
                    print(f"    Skipping metric '{metric}' for {op_type}")
                    continue

                metric_label, m_cmap, m_vmin, m_vmax = metric_configs.get(
                    metric, (metric.replace("_", " ").title(), cmap, None, None)
                )
                m_fmt = ".0f" if metric in ("throughput", "mean_latency", "p99_latency", "p95_latency", "p50_latency") else fmt

                if not xheatmap_only or metric == "success_rate":
                    _create_single_heatmap(
                        type_results, x_param, y_param, metric,
                        title=f"{op_label} {metric_label}{title_suffix}",
                        output_path=os.path.join(output_dir, f"heatmap_{prefix}_{metric}.png"),
                        cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, fmt=m_fmt,
                        figsize=figsize, dpi=dpi,
                    )

                if metric != "success_rate":
                    _create_single_heatmap(
                        type_results, x_param, y_param, metric,
                        title=f"{op_label} {metric_label}{title_suffix}",
                        output_path=os.path.join(output_dir, f"xheatmap_{prefix}_{metric}.png"),
                        cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, fmt=m_fmt,
                        figsize=figsize, dpi=dpi,
                        success_rate_data=type_results,
                        implausible_threshold=implausible_threshold,
                        unreliable_threshold=unreliable_threshold,
                    )


# ---------------------------------------------------------------------------
# Operation type analysis (moved from scripts/analyze_operation_types.py)
# ---------------------------------------------------------------------------

def _analyze_op_type_experiments(experiments_dir: str, pattern: str,
                                  group_by: str = None,
                                  experiments_cache: dict = None) -> pd.DataFrame:
    """Extract per-operation-type metrics from experiment directories and consolidated.

    Args:
        experiments_dir: Base directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        group_by: Additional config parameter to extract (e.g. 'catalog_service_latency_ms').
                  If None, defaults to 'fa_ratio'.
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
    """
    records = []
    experiments = experiments_cache if experiments_cache is not None else scan_all_experiments(experiments_dir, pattern)
    consolidated_path = os.path.join(experiments_dir, 'consolidated.parquet')
    reader = _ConsolidatedReader.open(consolidated_path)

    for dir_key, exp_info in experiments.items():
        cfg = exp_info['config']

        txn = cfg.get("transaction", {})
        fa_ratio = txn.get("operation_types", {}).get("fast_append", 0.5)
        scale = txn.get("inter_arrival", {}).get("scale", 100.0)
        num_tables = _extract_param_value(cfg, "num_tables") or 1

        # Extract additional group_by parameter if requested
        group_val = None
        if group_by and group_by not in ("fa_ratio", "inter_arrival_scale"):
            group_val = _extract_param_value(cfg, group_by)

        if exp_info['dir'] is None:
            # Consolidated-only: load from consolidated parquet
            if reader is None:
                continue
            df = _load_from_consolidated(reader, exp_info)
            if df is None or len(df) == 0:
                continue

            for seed_val in df['seed'].unique() if 'seed' in df.columns else [0]:
                seed_df = df[df['seed'] == seed_val] if 'seed' in df.columns else df

                for op_type in seed_df["operation_type"].dropna().unique():
                    op_df = seed_df[seed_df["operation_type"] == op_type]
                    if len(op_df) == 0:
                        continue

                    committed = op_df[op_df["status"].str.lower() == "committed"]

                    record = {
                        "seed": str(seed_val),
                        "fa_ratio": fa_ratio,
                        "fast_append_ratio": fa_ratio,
                        "num_tables": num_tables,
                        "inter_arrival_scale": scale,
                        "operation_type": op_type,
                        "total": len(op_df),
                        "committed": len(committed),
                        "aborted": len(op_df) - len(committed),
                        "success_rate": len(committed) / len(op_df) * 100 if len(op_df) > 0 else 0,
                        "mean_latency": committed["commit_latency"].mean() if len(committed) > 0 else np.nan,
                        "p95_latency": committed["commit_latency"].quantile(0.95) if len(committed) > 0 else np.nan,
                        "mean_retries": committed["n_retries"].mean() if len(committed) > 0 else np.nan,
                    }
                    if group_by and group_by not in ("fa_ratio", "inter_arrival_scale"):
                        record[group_by] = group_val
                    records.append(record)
        else:
            exp_dir = Path(exp_info['dir'])
            for seed_dir in exp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                results_path = seed_dir / "results.parquet"
                if not results_path.exists():
                    continue

                df = pd.read_parquet(results_path)

                for op_type in df["operation_type"].dropna().unique():
                    op_df = df[df["operation_type"] == op_type]
                    if len(op_df) == 0:
                        continue

                    committed = op_df[op_df["status"].str.lower() == "committed"]

                    record = {
                        "seed": seed_dir.name,
                        "fa_ratio": fa_ratio,
                        "fast_append_ratio": fa_ratio,
                        "num_tables": num_tables,
                        "inter_arrival_scale": scale,
                        "operation_type": op_type,
                        "total": len(op_df),
                        "committed": len(committed),
                        "aborted": len(op_df) - len(committed),
                        "success_rate": len(committed) / len(op_df) * 100 if len(op_df) > 0 else 0,
                        "mean_latency": committed["commit_latency"].mean() if len(committed) > 0 else np.nan,
                        "p95_latency": committed["commit_latency"].quantile(0.95) if len(committed) > 0 else np.nan,
                        "mean_retries": committed["n_retries"].mean() if len(committed) > 0 else np.nan,
                    }
                    if group_by and group_by not in ("fa_ratio", "inter_arrival_scale"):
                        record[group_by] = group_val
                    records.append(record)

    return pd.DataFrame(records)


def generate_operation_type_plots(base_dir: str, pattern: str, output_dir: str,
                                  load_levels: list = None, config: dict = None,
                                  group_by: str = None,
                                  experiments_cache: dict = None):
    """Generate per-operation-type comparison plots as a 2x2 grid.

    Layout:
        Top-left:     FA success rate vs load
        Top-right:    VO success rate vs load
        Bottom-left:  FA mean latency vs load
        Bottom-right: VO mean latency vs load

    X-axis is inter_arrival_scale (log scale, reversed — high load on left).
    Lines are per group_by value (fa_ratio for exp2, catalog_service_latency_ms for exp3b).

    Args:
        base_dir: Directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        output_dir: Directory to write plots.
        load_levels: Unused (kept for backward compatibility). All available loads are plotted.
        config: Optional plotting config overrides.
        group_by: Parameter to group lines by. Defaults to 'fa_ratio'.
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
    """
    config = config or {}
    group_by = group_by or "fa_ratio"
    os.makedirs(output_dir, exist_ok=True)
    dpi = config.get("dpi", DEFAULT_DPI)
    figsize = config.get("figsize", [12, 7.5])

    print(f"Analyzing operation types for {pattern}...")
    df = _analyze_op_type_experiments(base_dir, pattern, group_by=group_by,
                                      experiments_cache=experiments_cache)
    if len(df) == 0:
        print(f"  No operation type data found for {pattern}")
        return

    # Apply filters from config (e.g. num_tables==1, fast_append_ratio==0.9)
    filters = config.get("filters", [])
    if filters:
        df = apply_filters(df, filters)
        if len(df) == 0:
            print(f"  No data after applying filters {filters}")
            return

    # Aggregate across seeds
    group_cols = [group_by, "inter_arrival_scale", "operation_type"]
    # Ensure group_by column exists; for fa_ratio it's always there
    if group_by not in df.columns:
        print(f"  group_by column '{group_by}' not found in data")
        return

    agg = df.groupby(group_cols).agg({
        "total": "mean",
        "committed": "mean",
        "aborted": "mean",
        "success_rate": "mean",
        "mean_latency": "mean",
        "p95_latency": "mean",
        "mean_retries": "mean",
    }).reset_index()
    agg["success_rate"] = agg["committed"] / agg["total"] * 100

    # Save detailed data
    csv_path = os.path.join(output_dir, "operation_type_metrics.csv")
    agg.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Get group values and sort them
    group_values = sorted(agg[group_by].unique())

    # Color map for group values
    # Colorblind-friendly palette (Tol's qualitative scheme)
    _cb_palette = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
                   '#DDCC77', '#CC6677', '#882255', '#AA4499', '#DDDDDD']
    colors = {val: _cb_palette[i % len(_cb_palette)] for i, val in enumerate(group_values)}
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Format group label for legend
    group_label = group_by.replace("_", " ").title()

    # 2x2 grid: [FA success, VO success] / [FA latency, VO latency]
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    panels = [
        (0, 0, "fast_append", "success_rate", "FA Success Rate (%)", False),
        (0, 1, "validated_overwrite", "success_rate", "VO Success Rate (%)", False),
        (1, 0, "fast_append", "mean_latency", "FA Mean Commit Latency (ms)", True),
        (1, 1, "validated_overwrite", "mean_latency", "VO Mean Commit Latency (ms)", True),
    ]

    for row, col, op_type, metric, ylabel, use_log_y in panels:
        ax = axes[row, col]
        op_data = agg[agg["operation_type"] == op_type]

        has_data = False
        for i, gval in enumerate(group_values):
            # Skip group values with no transactions of this type
            # fa_ratio=0.0 has no FA transactions; fa_ratio=1.0 has no VO transactions
            if group_by == "fa_ratio":
                if op_type == "fast_append" and gval == 0.0:
                    continue
                if op_type == "validated_overwrite" and gval == 1.0:
                    continue

            subset = op_data[op_data[group_by] == gval].sort_values("inter_arrival_scale")
            if len(subset) == 0:
                continue

            marker = markers[i % len(markers)]
            label = f"{group_label}={gval}"
            ax.plot(subset["inter_arrival_scale"], subset[metric],
                    color=colors[gval], marker=marker, label=label,
                    linewidth=2, markersize=6, alpha=0.8)
            has_data = True

        ax.set_xlabel("Inter-Arrival Scale (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.invert_xaxis()  # High load (small scale) on left
        if use_log_y and has_data:
            ax.set_yscale("log")
        if metric == "success_rate":
            ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        if has_data:
            ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Per-Operation-Type Metrics vs Load (grouped by {group_label})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = os.path.join(output_dir, "op_type_vs_load.png")
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Per-table breakdown analysis (for Zipf experiments)
# ---------------------------------------------------------------------------

def _extract_table_ranks(df: pd.DataFrame) -> pd.Series:
    """Extract target table ID from write_table_ids and map to rank by frequency.

    Rank 0 = most frequent (hottest under Zipf).
    Returns a Series of integer ranks aligned with df's index.
    """
    # Extract first table ID from list column
    table_ids = df["write_table_ids"].apply(
        lambda x: x[0] if x is not None and len(x) > 0 else None
    )
    # Rank by frequency (most frequent = rank 0)
    counts = table_ids.value_counts()
    rank_map = {tid: rank for rank, tid in enumerate(counts.index)}
    return table_ids.map(rank_map)


def _apply_warmup_cooldown(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to steady-state using 10% warmup/cooldown."""
    duration = df["t_commit"].max() - df["t_submit"].min()
    if duration <= 0:
        return df
    warmup = duration * 0.1
    t_min = df["t_submit"].min() + warmup
    t_max = df["t_commit"].max() - duration * 0.1
    return df[(df["t_submit"] >= t_min) & (df["t_commit"] <= t_max)]


def _analyze_per_table_experiments(
    experiments_dir: str, pattern: str,
    filters: list = None, load_levels: list = None,
    num_tables_values: list = None,
    experiments_cache: dict = None,
) -> pd.DataFrame:
    """Extract per-table-rank metrics from experiment parquets.

    Returns a DataFrame with columns:
        num_tables, inter_arrival_scale, catalog_service_latency_ms,
        table_rank, operation_type, seed, write_share, success_rate,
        mean_retries, catalog_conflicts, tblptn_conflicts,
        mean_latency, p95_latency
    """
    records = []
    experiments = experiments_cache if experiments_cache is not None else scan_all_experiments(experiments_dir, pattern)
    consolidated_path = os.path.join(experiments_dir, 'consolidated.parquet')
    reader = _ConsolidatedReader.open(consolidated_path)

    for dir_key, exp_info in experiments.items():
        cfg = exp_info['config']

        num_tables = _extract_param_value(cfg, "num_tables")
        scale = _extract_param_value(cfg, "inter_arrival_scale")
        cas_latency = _extract_param_value(cfg, "catalog_service_latency_ms")

        # Apply config-level filters
        if num_tables_values and num_tables not in num_tables_values:
            continue
        if load_levels and scale not in load_levels:
            continue
        if filters:
            skip = False
            for filt in filters:
                val = None
                for op in ("==", "!=", "<=", ">=", "<", ">"):
                    if op in filt:
                        param, threshold = filt.split(op, 1)
                        param = param.strip()
                        threshold = float(threshold.strip())
                        val = _extract_param_value(cfg, param)
                        if val is None:
                            skip = True
                            break
                        if op == "==" and val != threshold:
                            skip = True
                        elif op == "!=" and val == threshold:
                            skip = True
                        elif op == "<" and val >= threshold:
                            skip = True
                        elif op == "<=" and val > threshold:
                            skip = True
                        elif op == ">" and val <= threshold:
                            skip = True
                        elif op == ">=" and val < threshold:
                            skip = True
                        break
                if skip:
                    break
            if skip:
                continue

        # Load data from disk or consolidated
        seed_dfs = []
        if exp_info['dir'] is None:
            if reader is None:
                continue
            df = _load_from_consolidated(reader, exp_info)
            if df is None or len(df) == 0:
                continue
            if "write_table_ids" not in df.columns:
                print(f"  Warning: {dir_key} missing write_table_ids, skipping")
                continue
            for seed_val in df['seed'].unique() if 'seed' in df.columns else [0]:
                seed_df = df[df['seed'] == seed_val] if 'seed' in df.columns else df
                seed_dfs.append((str(seed_val), seed_df))
        else:
            exp_dir = Path(exp_info['dir'])
            for seed_dir in exp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                results_path = seed_dir / "results.parquet"
                if not results_path.exists():
                    continue
                df = pd.read_parquet(results_path)
                if "write_table_ids" not in df.columns:
                    print(f"  Warning: {results_path} missing write_table_ids, skipping")
                    continue
                seed_dfs.append((seed_dir.name, df))

        for seed_name, seed_df in seed_dfs:
            # Apply warmup/cooldown
            seed_df = _apply_warmup_cooldown(seed_df)
            if len(seed_df) == 0:
                continue

            # Extract table ranks
            seed_df = seed_df.copy()
            seed_df["table_rank"] = _extract_table_ranks(seed_df)
            seed_df = seed_df.dropna(subset=["table_rank"])
            seed_df["table_rank"] = seed_df["table_rank"].astype(int)
            total_txns = len(seed_df)

            # Determine if mixed workload
            has_op_types = ("operation_type" in seed_df.columns and
                            seed_df["operation_type"].nunique() > 1)

            if has_op_types:
                group_cols = ["table_rank", "operation_type"]
            else:
                group_cols = "table_rank"

            for group_vals, group_df in seed_df.groupby(group_cols):
                if has_op_types:
                    trank, op_type = group_vals
                else:
                    trank = group_vals
                    op_type = "fast_append"

                committed = group_df[group_df["status"].str.lower() == "committed"]
                total = len(group_df)

                record = {
                    "num_tables": num_tables,
                    "inter_arrival_scale": scale,
                    "catalog_service_latency_ms": cas_latency,
                    "table_rank": trank,
                    "operation_type": op_type,
                    "seed": seed_name,
                    "write_share": total / total_txns * 100 if total_txns > 0 else 0,
                    "success_rate": len(committed) / total * 100 if total > 0 else 0,
                    "mean_retries": group_df["n_retries"].mean() if "n_retries" in group_df.columns else np.nan,
                    "catalog_conflicts": group_df["catalog_conflicts"].mean() if "catalog_conflicts" in group_df.columns else np.nan,
                    "tblptn_conflicts": group_df["tblptn_conflicts"].mean() if "tblptn_conflicts" in group_df.columns else np.nan,
                    "mean_latency": committed["commit_latency"].mean() if len(committed) > 0 else np.nan,
                    "p95_latency": committed["commit_latency"].quantile(0.95) if len(committed) > 0 else np.nan,
                }
                records.append(record)

    return pd.DataFrame(records)


def _plot_per_table_success_rate(agg: pd.DataFrame, output_dir: str,
                                 num_tables: int, scale: float,
                                 cas_latency: float, is_mixed: bool,
                                 figsize: tuple, dpi: int):
    """Plot 1: Per-table success rate bar chart."""
    subset = agg[
        (agg["num_tables"] == num_tables) &
        (agg["inter_arrival_scale"] == scale) &
        (agg["catalog_service_latency_ms"] == cas_latency)
    ].sort_values("table_rank")

    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=figsize)

    if is_mixed:
        op_types = sorted(subset["operation_type"].unique())
        n_ops = len(op_types)
        bar_width = 0.8 / n_ops
        _cb_palette = ['#332288', '#CC6677', '#44AA99', '#DDCC77']
        for i, op_type in enumerate(op_types):
            op_sub = subset[subset["operation_type"] == op_type].sort_values("table_rank")
            if len(op_sub) == 0:
                continue
            x = np.arange(len(op_sub))
            offset = (i - (n_ops - 1) / 2) * bar_width
            bars = ax.bar(x + offset, op_sub["success_rate"], bar_width * 0.9,
                          label=op_type.replace("_", " ").title(),
                          color=_cb_palette[i % len(_cb_palette)], alpha=0.85)
            # Annotate with write share
            for j, (_, row) in enumerate(op_sub.iterrows()):
                ax.text(x[j] + offset, row["success_rate"] + 1,
                        f'{row["write_share"]:.1f}%', ha='center', va='bottom',
                        fontsize=7, rotation=45)
        ax.set_xticks(np.arange(len(subset["table_rank"].unique())))
    else:
        ranks = subset["table_rank"].values
        sr = subset["success_rate"].values
        ws = subset["write_share"].values
        # Color gradient by write share
        norm = plt.Normalize(vmin=ws.min(), vmax=ws.max())
        cmap = plt.cm.YlOrRd
        colors = cmap(norm(ws))
        bars = ax.bar(np.arange(len(ranks)), sr, color=colors, alpha=0.85)
        # Annotate with write share
        for j in range(len(ranks)):
            ax.text(j, sr[j] + 1, f'{ws[j]:.1f}%', ha='center', va='bottom',
                    fontsize=8, rotation=45)
        ax.set_xticks(np.arange(len(ranks)))
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Write Share (%)", fontsize=10)

    ax.set_xticklabels(sorted(subset["table_rank"].unique()))
    ax.set_xlabel("Table Rank (0 = hottest)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Per-Table Success Rate\n"
        f"(N={num_tables}, load={scale}ms, CAS={cas_latency}ms)",
        fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis='y')
    if is_mixed:
        ax.legend(fontsize=10, loc="upper left")
    plt.tight_layout()

    fname = f"per_table_success_rate_t{num_tables}_s{int(scale)}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"  Saved: {path}")


def _plot_write_concentration(agg: pd.DataFrame, output_dir: str,
                              scale: float, cas_latency: float,
                              num_tables_values: list,
                              figsize: tuple, dpi: int):
    """Plot 2: Lorenz-style write concentration curve."""
    fig, ax = plt.subplots(figsize=figsize)

    _cb_palette = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933',
                   '#CC6677', '#882255', '#AA4499']

    has_data = False
    for i, nt in enumerate(sorted(num_tables_values)):
        subset = agg[
            (agg["num_tables"] == nt) &
            (agg["inter_arrival_scale"] == scale) &
            (agg["catalog_service_latency_ms"] == cas_latency)
        ]
        if len(subset) == 0:
            continue

        # Aggregate write_share across operation types per table rank
        # Sort ascending (least traffic first) for standard Lorenz curve
        ws_by_rank = subset.groupby("table_rank")["write_share"].sum().sort_values(ascending=True)
        ws_arr = ws_by_rank.values
        ws_arr = ws_arr / ws_arr.sum()  # Normalize to fractions

        # Lorenz curve: cumulative fraction of tables vs cumulative fraction of writes
        n = len(ws_arr)
        cum_tables = np.arange(1, n + 1) / n
        cum_writes = np.cumsum(ws_arr)

        # Prepend origin
        cum_tables = np.insert(cum_tables, 0, 0)
        cum_writes = np.insert(cum_writes, 0, 0)

        # Gini coefficient: 1 - 2 * area_under_lorenz
        area = np.trapz(cum_writes, cum_tables)
        gini = 1 - 2 * area

        ax.plot(cum_tables, cum_writes, marker='o', markersize=4,
                color=_cb_palette[i % len(_cb_palette)],
                linewidth=2, label=f"N={nt} (Gini={gini:.2f})")
        has_data = True

    if not has_data:
        plt.close()
        return

    # Diagonal reference line (perfect equality)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label="Uniform")

    ax.set_xlabel("Cumulative Fraction of Tables (ranked by traffic)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cumulative Fraction of Writes", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Write Concentration (Lorenz Curve)\n"
        f"(load={scale}ms, CAS={cas_latency}ms)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    fname = f"write_concentration_s{int(scale)}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"  Saved: {path}")


def _plot_conflict_type_by_table(agg: pd.DataFrame, output_dir: str,
                                 num_tables: int, scale: float,
                                 cas_latency: float, is_mixed: bool,
                                 figsize: tuple, dpi: int):
    """Plot 3: Conflict type stacked bar by table rank."""
    subset = agg[
        (agg["num_tables"] == num_tables) &
        (agg["inter_arrival_scale"] == scale) &
        (agg["catalog_service_latency_ms"] == cas_latency)
    ].sort_values("table_rank")

    if len(subset) == 0:
        return

    fig, ax = plt.subplots(figsize=figsize)

    if is_mixed:
        op_types = sorted(subset["operation_type"].unique())
        n_ops = len(op_types)
        bar_width = 0.8 / n_ops
        _cat_colors = ['#88CCEE', '#4477AA']  # catalog conflict colors per op type
        _tbl_colors = ['#CC6677', '#882255']   # tblptn conflict colors per op type

        for i, op_type in enumerate(op_types):
            op_sub = subset[subset["operation_type"] == op_type].sort_values("table_rank")
            if len(op_sub) == 0:
                continue
            x = np.arange(len(op_sub))
            offset = (i - (n_ops - 1) / 2) * bar_width
            op_label = op_type.replace("_", " ").title()

            cat_vals = op_sub["catalog_conflicts"].values
            tbl_vals = op_sub["tblptn_conflicts"].values

            ax.bar(x + offset, cat_vals, bar_width * 0.9,
                   label=f"{op_label} - Catalog",
                   color=_cat_colors[i % len(_cat_colors)], alpha=0.85)
            ax.bar(x + offset, tbl_vals, bar_width * 0.9, bottom=cat_vals,
                   label=f"{op_label} - Table/Ptn",
                   color=_tbl_colors[i % len(_tbl_colors)], alpha=0.85)
        ax.set_xticks(np.arange(len(subset["table_rank"].unique())))
    else:
        ranks = subset["table_rank"].values
        cat_vals = subset["catalog_conflicts"].values
        tbl_vals = subset["tblptn_conflicts"].values
        x = np.arange(len(ranks))

        ax.bar(x, cat_vals, color='#88CCEE', alpha=0.85, label="Catalog Conflicts")
        ax.bar(x, tbl_vals, bottom=cat_vals, color='#CC6677', alpha=0.85,
               label="Table/Partition Conflicts")
        ax.set_xticks(x)

    ax.set_xticklabels(sorted(subset["table_rank"].unique()))
    ax.set_xlabel("Table Rank (0 = hottest)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Conflicts per Transaction", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Conflict Type by Table Rank\n"
        f"(N={num_tables}, load={scale}ms, CAS={cas_latency}ms)",
        fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fname = f"conflict_type_by_table_t{num_tables}_s{int(scale)}.png"
    path = os.path.join(output_dir, fname)
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"  Saved: {path}")


def _analyze_workload_experiments(experiments_dir: str, pattern: str,
                                  filters: list = None,
                                  experiments_cache: dict = None) -> pd.DataFrame:
    """Extract per-operation-type metrics with provider and table count.

    Similar to _analyze_op_type_experiments but always extracts
    storage_provider and num_tables alongside fa_ratio and scale.
    Applies warmup/cooldown filtering (middle 80% of simulation).

    Args:
        experiments_dir: Base directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        filters: Optional list of filter expressions (e.g. ["num_tables==1"]).
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().

    Returns:
        DataFrame with columns: seed, storage_provider, num_tables, fa_ratio,
        inter_arrival_scale, operation_type, total, committed, success_rate,
        mean_latency, p95_latency, active_window_s
    """
    records = []
    experiments = experiments_cache if experiments_cache is not None else scan_all_experiments(experiments_dir, pattern)
    consolidated_path = os.path.join(experiments_dir, 'consolidated.parquet')
    reader = _ConsolidatedReader.open(consolidated_path)

    for dir_key, exp_info in experiments.items():
        cfg = exp_info['config']

        txn = cfg.get("transaction", {})
        fa_ratio = txn.get("operation_types", {}).get("fast_append", 0.5)
        scale = txn.get("inter_arrival", {}).get("scale", 100.0)
        provider = _extract_param_value(cfg, "storage_provider") or "unknown"
        num_tables = _extract_param_value(cfg, "num_tables") or 1

        def _process_seed_df(seed_df, seed_label):
            # Warmup/cooldown: keep middle 80%
            duration = seed_df["t_commit"].max() - seed_df["t_submit"].min()
            if duration <= 0:
                return
            warmup = duration * 0.1
            t_min = seed_df["t_submit"].min() + warmup
            t_max = seed_df["t_commit"].max() - warmup
            steady = seed_df[(seed_df["t_submit"] >= t_min) & (seed_df["t_commit"] <= t_max)]
            if len(steady) == 0:
                return
            active_window_s = (t_max - t_min) / 1000.0

            for op_type in steady["operation_type"].dropna().unique():
                op_df = steady[steady["operation_type"] == op_type]
                if len(op_df) == 0:
                    continue
                committed = op_df[op_df["status"].str.lower() == "committed"]
                records.append({
                    "seed": seed_label,
                    "storage_provider": provider,
                    "num_tables": num_tables,
                    "fa_ratio": fa_ratio,
                    "inter_arrival_scale": scale,
                    "operation_type": op_type,
                    "total": len(op_df),
                    "committed": len(committed),
                    "success_rate": len(committed) / len(op_df) * 100 if len(op_df) > 0 else 0,
                    "mean_latency": committed["commit_latency"].mean() if len(committed) > 0 else np.nan,
                    "p95_latency": committed["commit_latency"].quantile(0.95) if len(committed) > 0 else np.nan,
                    "active_window_s": active_window_s,
                })

        if exp_info['dir'] is None:
            if reader is None:
                continue
            df = _load_from_consolidated(reader, exp_info)
            if df is None or len(df) == 0:
                continue
            for seed_val in df['seed'].unique() if 'seed' in df.columns else [0]:
                seed_df = df[df['seed'] == seed_val] if 'seed' in df.columns else df
                _process_seed_df(seed_df, str(seed_val))
        else:
            exp_dir = Path(exp_info['dir'])
            for seed_dir in exp_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                    continue
                results_path = seed_dir / "results.parquet"
                if not results_path.exists():
                    continue
                df = pd.read_parquet(results_path)
                _process_seed_df(df, seed_dir.name)

    result = pd.DataFrame(records)

    if filters and len(result) > 0:
        result = apply_filters(result, filters)

    return result


def generate_workload_knee_table(base_dir: str, pattern: str, output_dir: str,
                                 config: dict = None,
                                 experiments_cache: dict = None):
    """Generate markdown table of usable (>95% success) commit rates per provider.

    For each (provider, num_tables, fa_ratio), finds the knee point — the
    lowest inter_arrival_scale where the binding constraint exceeds 95% success.
    Extracts total committed/sec and per-type mean latencies at that point.

    Outputs:
        workload_knee_table.md — one markdown section per num_tables value
        workload_knee_data.csv — full per-type metrics at every load point

    Args:
        base_dir: Directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        output_dir: Directory to write outputs.
        config: Optional config overrides (e.g. success_threshold).
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
    """
    config = config or {}
    success_threshold = config.get("success_threshold", 95.0)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing workload knees for {pattern}...")
    df = _analyze_workload_experiments(base_dir, pattern,
                                       filters=config.get("filters"),
                                       experiments_cache=experiments_cache)
    if len(df) == 0:
        print(f"  No workload data found for {pattern}")
        return

    # Aggregate across seeds: average metrics, sum counts
    group_cols = ["storage_provider", "num_tables", "fa_ratio",
                  "inter_arrival_scale", "operation_type"]
    agg = df.groupby(group_cols).agg({
        "total": "mean",
        "committed": "mean",
        "success_rate": "mean",
        "mean_latency": "mean",
        "p95_latency": "mean",
        "active_window_s": "mean",
    }).reset_index()
    # Recompute success_rate from per-seed averages for accuracy
    agg["success_rate"] = agg["committed"] / agg["total"] * 100

    # Save full CSV
    csv_path = os.path.join(output_dir, "workload_knee_data.csv")
    agg.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Compute throughput: total committed/sec across all op types per load point
    load_cols = ["storage_provider", "num_tables", "fa_ratio", "inter_arrival_scale"]
    throughput = agg.groupby(load_cols).agg(
        total_committed=("committed", "sum"),
        active_window_s=("active_window_s", "first"),
    ).reset_index()
    throughput["committed_per_sec"] = (
        throughput["total_committed"] / throughput["active_window_s"]
    )

    # Pivot op-type metrics for easy lookup
    fa_data = agg[agg["operation_type"] == "fast_append"].copy()
    vo_data = agg[agg["operation_type"].isin(
        ["validated_overwrite", "merge_append"]
    )].copy()

    # Find knee for each (provider, num_tables, fa_ratio)
    knee_records = []
    for (prov, nt, far), grp in throughput.groupby(
        ["storage_provider", "num_tables", "fa_ratio"]
    ):
        # Sort by inter_arrival_scale ascending (heaviest load first)
        grp = grp.sort_values("inter_arrival_scale", ascending=True)

        for _, row in grp.iterrows():
            ias = row["inter_arrival_scale"]

            # Determine binding constraint success rate
            if far == 1.0:
                constraint_df = fa_data[
                    (fa_data["storage_provider"] == prov) &
                    (fa_data["num_tables"] == nt) &
                    (fa_data["fa_ratio"] == far) &
                    (fa_data["inter_arrival_scale"] == ias)
                ]
            else:
                constraint_df = vo_data[
                    (vo_data["storage_provider"] == prov) &
                    (vo_data["num_tables"] == nt) &
                    (vo_data["fa_ratio"] == far) &
                    (vo_data["inter_arrival_scale"] == ias)
                ]

            if len(constraint_df) == 0:
                continue
            constraint_success = constraint_df["success_rate"].values[0]
            if constraint_success < success_threshold:
                continue

            # This is the knee — highest throughput meeting threshold
            fa_row = fa_data[
                (fa_data["storage_provider"] == prov) &
                (fa_data["num_tables"] == nt) &
                (fa_data["fa_ratio"] == far) &
                (fa_data["inter_arrival_scale"] == ias)
            ]
            vo_row = vo_data[
                (vo_data["storage_provider"] == prov) &
                (vo_data["num_tables"] == nt) &
                (vo_data["fa_ratio"] == far) &
                (vo_data["inter_arrival_scale"] == ias)
            ]

            fa_lat = fa_row["mean_latency"].values[0] / 1000.0 if len(fa_row) > 0 else np.nan
            vo_lat = vo_row["mean_latency"].values[0] / 1000.0 if len(vo_row) > 0 else np.nan

            knee_records.append({
                "storage_provider": prov,
                "num_tables": nt,
                "fa_ratio": far,
                "inter_arrival_scale": ias,
                "committed_per_sec": row["committed_per_sec"],
                "fa_mean_latency_s": fa_lat,
                "vo_mean_latency_s": vo_lat,
            })
            break  # Found knee for this combo

    if not knee_records:
        print("  No knee points found above threshold")
        return

    knee_df = pd.DataFrame(knee_records)

    # Provider display order
    provider_order = ["instant", "s3x", "s3", "azurex", "azure", "gcp"]
    knee_df["_prov_order"] = knee_df["storage_provider"].map(
        {p: i for i, p in enumerate(provider_order)}
    ).fillna(99)
    knee_df = knee_df.sort_values(["num_tables", "_prov_order", "fa_ratio"])

    # Build markdown
    lines = ["# Workload Knee Table", "",
             f"Highest total committed/sec where binding constraint > {success_threshold:.0f}% success.", ""]

    fa_ratios = sorted(knee_df["fa_ratio"].unique(), reverse=True)

    for nt in sorted(knee_df["num_tables"].unique()):
        nt_df = knee_df[knee_df["num_tables"] == nt]
        lines.append(f"## num_tables = {int(nt)}")
        lines.append("")

        # Header
        header = "| Provider"
        sep = "|----------"
        for far in fa_ratios:
            fa_pct = int(far * 100)
            vo_pct = 100 - fa_pct
            label = f"{fa_pct}/{vo_pct}"
            header += f" | {label} (c/s)"
            header += f" | {label} (FA/VO lat)"
            sep += "|-----------:" * 2
        header += " |"
        sep += "|"
        lines.append(header)
        lines.append(sep)

        for prov in provider_order:
            prov_df = nt_df[nt_df["storage_provider"] == prov]
            if len(prov_df) == 0:
                continue
            row_str = f"| {prov}"
            for far in fa_ratios:
                match = prov_df[prov_df["fa_ratio"] == far]
                if len(match) == 0:
                    row_str += " | — | — "
                else:
                    m = match.iloc[0]
                    row_str += f" | {m['committed_per_sec']:.1f}"
                    fa_s = f"{m['fa_mean_latency_s']:.2f}s" if not np.isnan(m['fa_mean_latency_s']) else "—"
                    vo_s = f"{m['vo_mean_latency_s']:.1f}s" if not np.isnan(m['vo_mean_latency_s']) else "—"
                    row_str += f" | {fa_s} / {vo_s}"
            row_str += " |"
            lines.append(row_str)

        lines.append("")

    md_path = os.path.join(output_dir, "workload_knee_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {md_path}")

    # Plot: committed/sec vs num_tables, one subplot per fa_ratio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = config.get("dpi", DEFAULT_DPI)
    n_ratios = len(fa_ratios)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5 * n_ratios, 5), squeeze=False)

    provider_colors = {
        "instant": "#888888", "s3x": "#1f77b4", "s3": "#ff7f0e",
        "azurex": "#2ca02c", "azure": "#d62728", "gcp": "#9467bd",
    }
    markers = ["o", "s", "D", "^", "v", "P"]

    for col_idx, far in enumerate(fa_ratios):
        ax = axes[0, col_idx]
        fa_pct = int(far * 100)
        vo_pct = 100 - fa_pct
        ax.set_title(f"FA/VO = {fa_pct}/{vo_pct}")

        ratio_df = knee_df[knee_df["fa_ratio"] == far]
        for i, prov in enumerate(provider_order):
            prov_data = ratio_df[ratio_df["storage_provider"] == prov].sort_values("num_tables")
            if len(prov_data) == 0:
                continue
            ax.plot(prov_data["num_tables"], prov_data["committed_per_sec"],
                    color=provider_colors.get(prov, "#333333"),
                    marker=markers[i % len(markers)], label=prov,
                    linewidth=2, markersize=6)

        ax.set_xlabel("Number of Tables")
        ax.set_ylabel("Committed / sec")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    plt.suptitle(f"Usable Throughput vs Table Count (>{success_threshold:.0f}% success)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "workload_knee_vs_tables.png")
    plt.savefig(plot_path, dpi=dpi)
    plt.close()
    print(f"  Saved: {plot_path}")


def generate_per_table_plots(base_dir: str, pattern: str, output_dir: str,
                              config: dict = None,
                              experiments_cache: dict = None):
    """Generate per-table-rank breakdown plots for Zipf experiments.

    Produces:
        - per_table_success_rate_t{N}_s{scale}.png (bar chart per operating point)
        - write_concentration_s{scale}.png (Lorenz curves)
        - conflict_type_by_table_t{N}_s{scale}.png (stacked bar)
        - per_table_data.csv (raw data)

    Args:
        base_dir: Directory containing experiment results.
        pattern: Glob pattern to match experiment directories.
        output_dir: Directory to write plots.
        config: Optional plotting config overrides with keys:
            filters, load_levels, num_tables_values.
        experiments_cache: Pre-scanned experiments dict from scan_all_experiments().
    """
    config = config or {}
    os.makedirs(output_dir, exist_ok=True)
    dpi = config.get("dpi", DEFAULT_DPI)
    figsize = tuple(config.get("figsize", [14, 8]))

    filters = config.get("filters", None)
    load_levels = config.get("load_levels", None)
    num_tables_values = config.get("num_tables_values", None)

    # Convert to float for matching
    if load_levels:
        load_levels = [float(x) for x in load_levels]
    if num_tables_values:
        num_tables_values = [int(x) for x in num_tables_values]

    print(f"Analyzing per-table breakdown for {pattern}...")
    df = _analyze_per_table_experiments(
        base_dir, pattern,
        filters=filters, load_levels=load_levels,
        num_tables_values=num_tables_values,
        experiments_cache=experiments_cache,
    )
    if len(df) == 0:
        print(f"  No per-table data found for {pattern}")
        return

    # Aggregate across seeds
    group_cols = ["num_tables", "inter_arrival_scale", "catalog_service_latency_ms",
                  "table_rank", "operation_type"]
    agg = df.groupby(group_cols).agg({
        "write_share": "mean",
        "success_rate": "mean",
        "mean_retries": "mean",
        "catalog_conflicts": "mean",
        "tblptn_conflicts": "mean",
        "mean_latency": "mean",
        "p95_latency": "mean",
    }).reset_index()

    # Save CSV
    csv_path = os.path.join(output_dir, "per_table_data.csv")
    agg.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    # Detect mixed workload
    is_mixed = agg["operation_type"].nunique() > 1

    # Get unique operating points
    nt_values = sorted(agg["num_tables"].unique())
    scale_values = sorted(agg["inter_arrival_scale"].unique())
    cas_values = sorted(agg["catalog_service_latency_ms"].unique())

    # Typically there's one CAS value after filtering
    for cas_lat in cas_values:
        # Plot 1 & 3: Per-table charts for each (num_tables, load) combo
        for nt in nt_values:
            for scale in scale_values:
                _plot_per_table_success_rate(
                    agg, output_dir, nt, scale, cas_lat, is_mixed, figsize, dpi)
                _plot_conflict_type_by_table(
                    agg, output_dir, nt, scale, cas_lat, is_mixed, figsize, dpi)

        # Plot 2: Lorenz curves (one per load level, all num_tables)
        for scale in scale_values:
            _plot_write_concentration(
                agg, output_dir, scale, cas_lat, nt_values, figsize, dpi)


def cli():
    global CONFIG

    parser = argparse.ArgumentParser(
        description="Saturation analysis for baseline experiments"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to TOML configuration file (optional)"
    )
    parser.add_argument(
        "-i", "--input-dir",
        help="Base directory containing experiment results (overrides config)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory for plots (overrides config)"
    )
    parser.add_argument(
        "-p", "--pattern",
        help="Pattern to match experiment directories (overrides config)"
    )
    parser.add_argument(
        "--group-by",
        help="Parameter to group by in plots (overrides config)"
    )
    parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        help="Filter experiments by parameter (e.g., 't_cas_mean==50'). Can be specified multiple times."
    )

    args = parser.parse_args()

    # Load configuration
    CONFIG = load_config(args.config)

    # CLI arguments override config
    input_dir = args.input_dir if args.input_dir else CONFIG['paths']['input_dir']
    output_dir = args.output_dir if args.output_dir else CONFIG['paths']['output_dir']
    pattern = args.pattern if args.pattern else CONFIG['paths']['pattern']
    group_by = args.group_by if args.group_by else CONFIG['analysis'].get('group_by')

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build experiment index
    print(f"Scanning {input_dir} for pattern '{pattern}'...")
    index_df = build_experiment_index(input_dir, pattern)

    print(f"\nBuilt index with {len(index_df)} experiments")
    print(f"Parameters found: {list(index_df.columns)}")

    # Apply filters if specified
    if args.filters:
        print(f"\nApplying {len(args.filters)} filter(s)...")
        index_df = apply_filters(index_df, args.filters)
        print(f"After filtering: {len(index_df)} experiments remain\n")

    # Save index
    index_filename = CONFIG['output']['files']['experiment_index']
    index_path = os.path.join(output_dir, index_filename)
    save_experiment_index(index_df, index_path)

    # Generate plots and tables
    print("\nGenerating plots and tables...")

    # Latency vs Throughput
    plot_filename = CONFIG['output']['files']['latency_vs_throughput_plot']
    table_filename = CONFIG['output']['files']['latency_vs_throughput_table']
    plot_path = os.path.join(output_dir, plot_filename)
    table_path = os.path.join(output_dir, table_filename)
    plot_latency_vs_throughput(
        index_df, plot_path,
        title="Commit Latency vs Throughput",
        group_by=group_by
    )
    generate_latency_vs_throughput_table(
        index_df, table_path,
        group_by=group_by
    )

    # Sustainable Throughput (99%+ success region only)
    plot_filename = CONFIG['output']['files']['sustainable_throughput_plot']
    plot_path = os.path.join(output_dir, plot_filename)
    plot_sustainable_throughput(
        index_df, plot_path,
        title="Sustainable Throughput",
        success_threshold=99.0,
        group_by=group_by
    )

    # Success rate vs load (if inter_arrival_scale present)
    if 'inter_arrival_scale' in index_df.columns:
        plot_filename = CONFIG['output']['files']['success_vs_load_plot']
        plot_path = os.path.join(output_dir, plot_filename)
        plot_success_rate_vs_load(index_df, plot_path)
        # No separate table for success_vs_load

    # Success rate vs throughput
    plot_filename = CONFIG['output']['files']['success_vs_throughput_plot']
    plot_path = os.path.join(output_dir, plot_filename)
    plot_success_rate_vs_throughput(
        index_df, plot_path,
        title="Transaction Success Rate vs Throughput",
        group_by=group_by
    )

    # Overhead vs throughput
    plot_filename = CONFIG['output']['files']['overhead_vs_throughput_plot']
    table_filename = CONFIG['output']['files']['overhead_vs_throughput_table']
    plot_path = os.path.join(output_dir, plot_filename)
    table_path = os.path.join(output_dir, table_filename)
    plot_overhead_vs_throughput(
        index_df, plot_path,
        title="Commit Overhead vs Throughput",
        group_by=group_by
    )
    generate_overhead_table(
        index_df, table_path,
        group_by=group_by
    )

    # Commit rate over time
    plot_filename = CONFIG['output']['files']['commit_rate_over_time_plot']
    plot_path = os.path.join(output_dir, plot_filename)
    plot_commit_rate_over_time(
        input_dir, pattern, plot_path,
        title="Commit Rate Over Time"
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    cli()
