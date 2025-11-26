#!/usr/bin/env python
"""
Saturation analysis module for baseline experiments.

This module generates the latency vs throughput graphs described in ANALYSIS_PLAN.md.
It reads experiment directories, extracts parameters from cfg.toml files,
aggregates results across multiple seeds, and produces visualization.
"""

import argparse
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
            'dpi': 300,
            'bbox_inches': 'tight',
            'figsize': {
                'latency_vs_throughput': [12, 8],
                'success_vs_load': [16, 6],
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
                'enabled': True,
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
                'commit_rate_over_time_plot': 'commit_rate_over_time.png'
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

        experiments[exp_dir] = {
            'config': config,
            'seeds': seed_dirs,
            'label': label,
            'hash': exp_hash,
            'dir': exp_dir
        }

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
        params['num_groups'] = config['catalog'].get('num_groups', None)

    # Extract transaction config
    if 'transaction' in config:
        params['real_conflict_probability'] = config['transaction'].get('real_conflict_probability', 0.0)

    # Extract simulation duration
    if 'simulation' in config:
        params['duration_ms'] = config['simulation'].get('duration_ms', None)

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


def load_and_aggregate_results_consolidated(exp_info: Dict, consolidated_path: str = 'experiments/consolidated.parquet') -> pd.DataFrame:
    """
    Load experiment results from consolidated parquet file using predicate pushdown.

    This is a memory-efficient alternative to load_and_aggregate_results() that uses
    the consolidated parquet file with predicate pushdown to only load relevant data.

    Args:
        exp_info: Experiment information dict with 'label', 'hash', 'seeds', and 'config'
        consolidated_path: Path to consolidated parquet file

    Returns:
        DataFrame with aggregated statistics across all seeds (warmup/cooldown excluded)
    """
    # Compute warmup and cooldown durations for this experiment
    warmup_ms = compute_warmup_duration(exp_info['config'])
    cooldown_ms = compute_cooldown_duration(exp_info['config'])

    # Get simulation duration to compute cooldown threshold
    sim_duration_ms = exp_info['config'].get('simulation', {}).get('duration_ms', 3600000)
    cooldown_start_ms = sim_duration_ms - cooldown_ms

    # Extract exp_name and exp_hash from exp_info
    exp_name = exp_info['label']
    exp_hash = exp_info['hash']

    # Use predicate pushdown to load only this experiment's data
    # Parquet will skip entire row groups that don't match these filters
    filters = [
        ('exp_name', '==', exp_name),
        ('exp_hash', '==', exp_hash),
        # Pushdown t_submit filter for warmup/cooldown
        ('t_submit', '>=', warmup_ms),
        ('t_submit', '<', cooldown_start_ms)
    ]

    try:
        df = pd.read_parquet(consolidated_path, filters=filters)
    except FileNotFoundError:
        print(f"Warning: Consolidated file not found at {consolidated_path}, falling back to individual files")
        return load_and_aggregate_results(exp_info)

    if len(df) == 0:
        return None

    # Drop the consolidated-specific columns (we don't need them for analysis)
    df = df.drop(columns=['exp_name', 'exp_hash', 'config'])

    # Report filtering statistics
    # Note: With predicate pushdown, we don't know the pre-filter count, so we approximate
    total_txns_after_filter = len(df)
    active_window_ms = cooldown_start_ms - warmup_ms

    # Estimate total txns (assuming uniform distribution over time)
    duration_ratio = sim_duration_ms / active_window_ms if active_window_ms > 0 else 1
    estimated_total = int(total_txns_after_filter * duration_ratio)
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
    experiments = scan_experiment_directories(base_dir, pattern)

    if not experiments:
        raise ValueError(f"No experiments found in {base_dir} matching {pattern}")

    print(f"Found {len(experiments)} experiment directories")

    rows = []

    for exp_dir, exp_info in experiments.items():
        # Extract parameters
        params = extract_key_parameters(exp_info['config'])

        # Load and aggregate results
        print(f"Processing {exp_info['label']}-{exp_info['hash']}...", end='')

        # Use consolidated file if enabled, otherwise use individual files
        use_consolidated = CONFIG.get('analysis', {}).get('use_consolidated', False)
        if use_consolidated:
            consolidated_path = CONFIG.get('paths', {}).get('consolidated_file', 'experiments/consolidated.parquet')
            df = load_and_aggregate_results_consolidated(exp_info, consolidated_path)
        else:
            df = load_and_aggregate_results(exp_info)

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
    group_by: str = None
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

    # Mark saturation point (configurable success rate threshold)
    sat_config = CONFIG.get('plots', {}).get('saturation', {})
    sat_enabled = sat_config.get('enabled', True)
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved latency vs throughput plot to {output_path}")
    plt.close()


def plot_success_rate_vs_load(
    index_df: pd.DataFrame,
    output_path: str,
    title: str = "Success Rate vs Offered Load"
):
    """Plot success rate and throughput vs offered load (inter-arrival time)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

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
    sat_enabled = sat_config.get('enabled', True)
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
               marker='o', linewidth=3, markersize=10, color='#2E86AB')

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
    sat_enabled = sat_config.get('enabled', True)
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved overhead vs throughput plot to {output_path}")
    plt.close()


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
    experiments = scan_experiment_directories(base_dir, pattern)

    if not experiments:
        print(f"No experiments found for commit rate plot")
        return

    # Get warmup duration for marking on plot
    first_exp = list(experiments.values())[0]
    warmup_ms = compute_transient_period_duration(first_exp['config'])
    warmup_min = warmup_ms / 60000

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))

    for (exp_dir, exp_info), color in zip(experiments.items(), colors):
        # Load aggregated results
        use_consolidated = CONFIG.get('analysis', {}).get('use_consolidated', False)
        if use_consolidated:
            consolidated_path = CONFIG.get('paths', {}).get('consolidated_file', 'experiments/consolidated.parquet')
            df = load_and_aggregate_results_consolidated(exp_info, consolidated_path)
        else:
            df = load_and_aggregate_results(exp_info)

        if df is None or len(df) == 0:
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

    ax.set_xlabel('Time (minutes)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Commit Rate (commits/sec)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
                        decimals=1
                    )
                    p95 = format_value_with_stddev(
                        row['p95_commit_latency']/1000,
                        row.get('p95_commit_latency_std')/1000 if has_stddev and row.get('p95_commit_latency_std') else None,
                        decimals=1
                    )
                    p99 = format_value_with_stddev(
                        row['p99_commit_latency']/1000,
                        row.get('p99_commit_latency_std')/1000 if has_stddev and row.get('p99_commit_latency_std') else None,
                        decimals=1
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
                    decimals=1
                )
                p95 = format_value_with_stddev(
                    row['p95_commit_latency']/1000,
                    row.get('p95_commit_latency_std')/1000 if has_stddev and row.get('p95_commit_latency_std') else None,
                    decimals=1
                )
                p99 = format_value_with_stddev(
                    row['p99_commit_latency']/1000,
                    row.get('p99_commit_latency_std')/1000 if has_stddev and row.get('p99_commit_latency_std') else None,
                    decimals=1
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


def save_experiment_index(index_df: pd.DataFrame, output_path: str):
    """Save experiment index to CSV for reference."""
    # Handle empty dataframes
    if index_df.empty:
        # Create empty DataFrame with expected column headers
        columns = [
            'label', 'hash', 'num_seeds',
            'inter_arrival_scale', 'num_tables', 'num_groups',
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
        'inter_arrival_scale', 'num_tables', 'num_groups',
        'real_conflict_probability',
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


def cli():
    global CONFIG

    parser = argparse.ArgumentParser(
        description="Saturation analysis for baseline experiments"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to analysis.toml configuration file"
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
