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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomli


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
    K_MIN_CYCLES = 5  # Number of transaction cycles for steady-state (increased from 3 to eliminate gradual decline)
    MIN_PERIOD_MS = 5 * 60 * 1000  # 5 minutes absolute minimum
    MAX_PERIOD_MS = 15 * 60 * 1000  # 15 minutes maximum

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


def compute_aggregate_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute aggregate statistics for an experiment.

    Returns statistics aggregated across all seeds.
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

    # Calculate duration (in seconds)
    duration_s = df['t_submit'].max() / 1000.0 if len(df) > 0 else 1.0

    # Compute overhead percentage for committed transactions
    # Overhead = (commit_latency / total_latency) * 100
    # This represents the percentage of time spent in commit protocol (retries, backoff, manifest I/O)
    # vs total transaction time (runtime + commit)
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
        'throughput': len(committed) / duration_s,  # commits per second
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
        df = load_and_aggregate_results(exp_info)

        if df is None:
            print(" no data")
            continue

        stats = compute_aggregate_statistics(df)

        if stats is None:
            print(" no statistics")
            continue

        print(f" {stats['committed']}/{stats['total_txns']} committed")

        # Combine into single row
        row = {
            'exp_dir': exp_dir,
            'label': exp_info['label'],
            'hash': exp_info['hash'],
            'num_seeds': len(exp_info['seeds']),
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

    # Mark saturation point (50% success rate)
    saturation_points = index_df[index_df['success_rate'] < 55]  # Some tolerance
    if len(saturation_points) > 0:
        sat_throughput = saturation_points['throughput'].max()
        ax.axvline(sat_throughput, color='red', linestyle='--',
                  linewidth=2, alpha=0.5, label='~50% Success Rate')
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
    ax1.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% Saturation Threshold')
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

    # Mark 50% success rate
    ax.axhline(50, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label='50% Saturation Threshold')

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


def save_experiment_index(index_df: pd.DataFrame, output_path: str):
    """Save experiment index to CSV for reference."""
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
    parser = argparse.ArgumentParser(
        description="Saturation analysis for baseline experiments"
    )
    parser.add_argument(
        "-i", "--input-dir",
        default="experiments",
        help="Base directory containing experiment results (default: experiments)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="exp2_*",
        help="Pattern to match experiment directories (default: exp2_*)"
    )
    parser.add_argument(
        "--group-by",
        help="Parameter to group by in plots (e.g., 'num_tables', 'real_conflict_probability')"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build experiment index
    print(f"Scanning {args.input_dir} for pattern '{args.pattern}'...")
    index_df = build_experiment_index(args.input_dir, args.pattern)

    print(f"\nBuilt index with {len(index_df)} experiments")
    print(f"Parameters found: {list(index_df.columns)}")

    # Save index
    index_path = os.path.join(args.output_dir, "experiment_index.csv")
    save_experiment_index(index_df, index_path)

    # Generate plots
    print("\nGenerating plots...")

    # Latency vs Throughput
    plot_path = os.path.join(args.output_dir, "latency_vs_throughput.png")
    plot_latency_vs_throughput(
        index_df, plot_path,
        title="Commit Latency vs Throughput",
        group_by=args.group_by
    )

    # Success rate vs load (if inter_arrival_scale present)
    if 'inter_arrival_scale' in index_df.columns:
        plot_path = os.path.join(args.output_dir, "success_rate_vs_load.png")
        plot_success_rate_vs_load(index_df, plot_path)

    # Success rate vs throughput
    plot_path = os.path.join(args.output_dir, "success_rate_vs_throughput.png")
    plot_success_rate_vs_throughput(
        index_df, plot_path,
        title="Transaction Success Rate vs Throughput",
        group_by=args.group_by
    )

    # Overhead vs throughput
    plot_path = os.path.join(args.output_dir, "overhead_vs_throughput.png")
    plot_overhead_vs_throughput(
        index_df, plot_path,
        title="Commit Overhead vs Throughput",
        group_by=args.group_by
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    cli()
