#!/usr/bin/env python3
"""
Warmup Period Validation Script

Validates that the chosen warmup period is sufficient by visualizing
how metrics evolve over the simulation duration.

Usage:
    python -m icecap.warmup_validation <experiment_dir> [--window-size SECONDS]

Example:
    python -m icecap.warmup_validation experiments/exp2_1_single_table_false-abc123/12345
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tomllib


def compute_warmup_duration(config: dict) -> float:
    """
    Compute warmup duration using transaction-runtime multiple approach.

    Args:
        config: Parsed TOML configuration dict

    Returns:
        Warmup duration in milliseconds
    """
    K_MIN_CYCLES = 3  # Number of transaction cycles for steady-state
    MIN_WARMUP_MS = 5 * 60 * 1000  # 5 minutes absolute minimum
    MAX_WARMUP_MS = 15 * 60 * 1000  # 15 minutes maximum

    # Get mean transaction runtime
    mean_runtime_ms = config.get("transaction", {}).get("runtime", {}).get("mean", 10000)

    # Calculate warmup based on transaction cycles
    warmup_ms = max(
        MIN_WARMUP_MS,
        min(
            K_MIN_CYCLES * mean_runtime_ms,
            MAX_WARMUP_MS
        )
    )

    return warmup_ms


def load_experiment_data(exp_dir: Path):
    """Load experiment configuration and results."""
    # Load config
    config_path = exp_dir.parent.parent / "cfg.toml"
    if not config_path.exists():
        # Try parent directory (in case exp_dir is the seed dir)
        config_path = exp_dir.parent / "cfg.toml"

    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find cfg.toml in {exp_dir}")

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Load results
    results_path = exp_dir / "results.parquet"
    if not results_path.exists():
        raise FileNotFoundError(f"Cannot find results.parquet in {exp_dir}")

    df = pd.read_parquet(results_path)

    return config, df


def compute_windowed_metrics(df: pd.DataFrame, window_size_ms: float):
    """
    Compute metrics in sliding time windows.

    Args:
        df: Transaction results DataFrame
        window_size_ms: Window size in milliseconds

    Returns:
        DataFrame with metrics per window
    """
    max_time = df['t_submit'].max()
    windows = []

    # Create non-overlapping windows
    start_time = 0
    while start_time < max_time:
        end_time = start_time + window_size_ms

        # Filter to window
        window_df = df[(df['t_submit'] >= start_time) & (df['t_submit'] < end_time)]

        if len(window_df) == 0:
            start_time = end_time
            continue

        # Compute metrics
        committed = window_df[window_df['status'] == 'committed']

        metrics = {
            'window_start_ms': start_time,
            'window_end_ms': end_time,
            'window_center_ms': (start_time + end_time) / 2,
            'total_txns': len(window_df),
            'committed': len(committed),
            'aborted': len(window_df) - len(committed),
            'success_rate': 100.0 * len(committed) / len(window_df) if len(window_df) > 0 else 0,
            'throughput': len(committed) / (window_size_ms / 1000.0),  # commits/sec
        }

        # Latency metrics (only for committed)
        if len(committed) > 0:
            metrics['mean_latency'] = committed['commit_latency'].mean()
            metrics['median_latency'] = committed['commit_latency'].median()
            metrics['p95_latency'] = committed['commit_latency'].quantile(0.95)
            metrics['p99_latency'] = committed['commit_latency'].quantile(0.99)
            metrics['mean_retries'] = committed['n_retries'].mean()
        else:
            metrics['mean_latency'] = np.nan
            metrics['median_latency'] = np.nan
            metrics['p95_latency'] = np.nan
            metrics['p99_latency'] = np.nan
            metrics['mean_retries'] = np.nan

        windows.append(metrics)
        start_time = end_time

    return pd.DataFrame(windows)


def plot_convergence(windowed_metrics: pd.DataFrame, warmup_ms: float, config: dict, output_path: str):
    """
    Plot metrics over time to visualize convergence.

    Args:
        windowed_metrics: DataFrame with metrics per window
        warmup_ms: Warmup duration in milliseconds
        config: Configuration dict for labels
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Convert time to minutes for readability
    time_min = windowed_metrics['window_center_ms'] / (60 * 1000)
    warmup_min = warmup_ms / (60 * 1000)

    # Get config parameters for title
    inter_arrival = config.get("transaction", {}).get("inter_arrival", {}).get("scale", "?")
    num_tables = config.get("catalog", {}).get("num_tables", "?")

    # Plot 1: Throughput over time
    ax = axes[0, 0]
    ax.plot(time_min, windowed_metrics['throughput'], 'o-', linewidth=2, markersize=6)
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Throughput (commits/sec)', fontsize=12)
    ax.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Success Rate over time
    ax = axes[0, 1]
    ax.plot(time_min, windowed_metrics['success_rate'], 'o-', linewidth=2, markersize=6, color='green')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.axhline(50, color='orange', linestyle=':', alpha=0.5, label='50% Threshold')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Mean Latency over time
    ax = axes[1, 0]
    ax.plot(time_min, windowed_metrics['mean_latency'], 'o-', linewidth=2, markersize=6, color='purple')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Mean Commit Latency (ms)', fontsize=12)
    ax.set_title('Mean Latency Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: P95 Latency over time
    ax = axes[1, 1]
    ax.plot(time_min, windowed_metrics['p95_latency'], 'o-', linewidth=2, markersize=6, color='orange')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('P95 Commit Latency (ms)', fontsize=12)
    ax.set_title('P95 Latency Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Overall title with config info
    fig.suptitle(f'Warmup Period Validation\n'
                 f'inter_arrival={inter_arrival}ms, num_tables={num_tables}, warmup={warmup_min:.1f} min',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to {output_path}")
    plt.close()


def print_convergence_analysis(windowed_metrics: pd.DataFrame, warmup_ms: float):
    """
    Print statistical analysis of convergence.

    Args:
        windowed_metrics: DataFrame with metrics per window
        warmup_ms: Warmup duration in milliseconds
    """
    # Split into warmup and active periods
    warmup_data = windowed_metrics[windowed_metrics['window_center_ms'] < warmup_ms]
    active_data = windowed_metrics[windowed_metrics['window_center_ms'] >= warmup_ms]

    if len(warmup_data) == 0 or len(active_data) == 0:
        print("Warning: Not enough data for convergence analysis")
        return

    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)

    print(f"\nWarmup Period: 0 - {warmup_ms/1000:.0f}s ({warmup_ms/60000:.1f} min)")
    print(f"Active Period: {warmup_ms/1000:.0f}s - {windowed_metrics['window_end_ms'].max()/1000:.0f}s")

    print(f"\nNumber of windows:")
    print(f"  Warmup: {len(warmup_data)}")
    print(f"  Active: {len(active_data)}")

    # Compare metrics
    metrics_to_compare = [
        ('throughput', 'Throughput (commits/sec)', 1),
        ('success_rate', 'Success Rate (%)', 1),
        ('mean_latency', 'Mean Latency (ms)', 0),
        ('p95_latency', 'P95 Latency (ms)', 0),
    ]

    print(f"\n{'Metric':<25} {'Warmup':<20} {'Active':<20} {'Change':<15}")
    print("-" * 80)

    for metric, label, decimals in metrics_to_compare:
        warmup_mean = warmup_data[metric].mean()
        active_mean = active_data[metric].mean()

        if warmup_mean != 0:
            change_pct = ((active_mean - warmup_mean) / warmup_mean) * 100
            change_str = f"{change_pct:+.1f}%"
        else:
            change_str = "N/A"

        print(f"{label:<25} {warmup_mean:>15.{decimals}f}   {active_mean:>15.{decimals}f}   {change_str:>12}")

    # Coefficient of variation (stability measure)
    print(f"\nCoefficient of Variation (lower = more stable):")
    print(f"{'Metric':<25} {'Warmup CV':<15} {'Active CV':<15}")
    print("-" * 55)

    for metric, label, _ in metrics_to_compare:
        if active_data[metric].mean() > 0:
            warmup_cv = (warmup_data[metric].std() / warmup_data[metric].mean()) * 100
            active_cv = (active_data[metric].std() / active_data[metric].mean()) * 100
            print(f"{label:<25} {warmup_cv:>10.1f}%      {active_cv:>10.1f}%")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Validate warmup period by visualizing metric convergence"
    )
    parser.add_argument(
        "experiment_dir",
        help="Path to experiment seed directory (e.g., experiments/exp2_1_*/12345)"
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=60.0,
        help="Window size in seconds for metric computation (default: 60s)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for plot (default: warmup_validation.png)"
    )

    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: Directory not found: {exp_dir}")
        sys.exit(1)

    print(f"Loading experiment data from {exp_dir}...")
    config, df = load_experiment_data(exp_dir)

    print(f"Loaded {len(df)} transactions")
    print(f"Duration: {df['t_submit'].max()/1000:.0f} seconds")

    # Compute warmup duration
    warmup_ms = compute_warmup_duration(config)
    print(f"\nComputed warmup period: {warmup_ms/1000:.0f}s ({warmup_ms/60000:.1f} min)")

    # Compute windowed metrics
    window_size_ms = args.window_size * 1000
    print(f"Computing metrics in {args.window_size}s windows...")
    windowed_metrics = compute_windowed_metrics(df, window_size_ms)
    print(f"Created {len(windowed_metrics)} windows")

    # Print convergence analysis
    print_convergence_analysis(windowed_metrics, warmup_ms)

    # Plot convergence
    if args.output:
        output_path = args.output
    else:
        output_path = f"warmup_validation_{exp_dir.parent.name}_{exp_dir.name}.png"

    plot_convergence(windowed_metrics, warmup_ms, config, output_path)

    print(f"\nValidation complete!")


if __name__ == "__main__":
    main()
