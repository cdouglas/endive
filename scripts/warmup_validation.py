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


def plot_convergence(windowed_metrics: pd.DataFrame, warmup_ms: float, cooldown_ms: float,
                    sim_duration_ms: float, config: dict, output_path: str):
    """
    Plot metrics over time to visualize convergence.

    Args:
        windowed_metrics: DataFrame with metrics per window
        warmup_ms: Warmup duration in milliseconds
        cooldown_ms: Cooldown duration in milliseconds
        sim_duration_ms: Simulation duration in milliseconds
        config: Configuration dict for labels
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Convert time to minutes for readability
    time_min = windowed_metrics['window_center_ms'] / (60 * 1000)
    warmup_min = warmup_ms / (60 * 1000)
    cooldown_start_min = (sim_duration_ms - cooldown_ms) / (60 * 1000)
    sim_duration_min = sim_duration_ms / (60 * 1000)

    # Get config parameters for title
    inter_arrival = config.get("transaction", {}).get("inter_arrival", {}).get("scale", "?")
    num_tables = config.get("catalog", {}).get("num_tables", "?")
    active_window_min = (sim_duration_ms - cooldown_ms - warmup_ms) / (60 * 1000)

    # Plot 1: Throughput over time
    ax = axes[0, 0]
    ax.plot(time_min, windowed_metrics['throughput'], 'o-', linewidth=2, markersize=6)
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.axvline(cooldown_start_min, color='blue', linestyle='--', linewidth=2, label='Cooldown Start')
    ax.axvspan(0, warmup_min, alpha=0.1, color='red', label='Warmup Period')
    ax.axvspan(cooldown_start_min, sim_duration_min, alpha=0.1, color='blue', label='Cooldown Period')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Throughput (commits/sec)', fontsize=12)
    ax.set_title('Throughput Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    # Plot 2: Success Rate over time
    ax = axes[0, 1]
    ax.plot(time_min, windowed_metrics['success_rate'], 'o-', linewidth=2, markersize=6, color='green')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.axvline(cooldown_start_min, color='blue', linestyle='--', linewidth=2, label='Cooldown Start')
    ax.axvspan(0, warmup_min, alpha=0.1, color='red')
    ax.axvspan(cooldown_start_min, sim_duration_min, alpha=0.1, color='blue')
    ax.axhline(50, color='orange', linestyle=':', alpha=0.5, label='50% Threshold')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    # Plot 3: Mean Latency over time
    ax = axes[1, 0]
    ax.plot(time_min, windowed_metrics['mean_latency'], 'o-', linewidth=2, markersize=6, color='purple')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.axvline(cooldown_start_min, color='blue', linestyle='--', linewidth=2, label='Cooldown Start')
    ax.axvspan(0, warmup_min, alpha=0.1, color='red')
    ax.axvspan(cooldown_start_min, sim_duration_min, alpha=0.1, color='blue')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Mean Commit Latency (ms)', fontsize=12)
    ax.set_title('Mean Latency Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    # Plot 4: P95 Latency over time
    ax = axes[1, 1]
    ax.plot(time_min, windowed_metrics['p95_latency'], 'o-', linewidth=2, markersize=6, color='orange')
    ax.axvline(warmup_min, color='red', linestyle='--', linewidth=2, label='Warmup End')
    ax.axvline(cooldown_start_min, color='blue', linestyle='--', linewidth=2, label='Cooldown Start')
    ax.axvspan(0, warmup_min, alpha=0.1, color='red')
    ax.axvspan(cooldown_start_min, sim_duration_min, alpha=0.1, color='blue')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('P95 Commit Latency (ms)', fontsize=12)
    ax.set_title('P95 Latency Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    # Overall title with config info
    fig.suptitle(f'Transient Period Validation\n'
                 f'inter_arrival={inter_arrival}ms, num_tables={num_tables}, '
                 f'warmup={warmup_min:.1f}min, cooldown={cooldown_ms/60000:.1f}min, active={active_window_min:.1f}min',
                 fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved convergence plot to {output_path}")
    plt.close()


def print_convergence_analysis(windowed_metrics: pd.DataFrame, warmup_ms: float, cooldown_start_ms: float):
    """
    Print statistical analysis of convergence.

    Args:
        windowed_metrics: DataFrame with metrics per window
        warmup_ms: Warmup duration in milliseconds
        cooldown_start_ms: Cooldown start time in milliseconds
    """
    # Split into three periods: warmup, active, cooldown
    warmup_data = windowed_metrics[windowed_metrics['window_center_ms'] < warmup_ms]
    active_data = windowed_metrics[(windowed_metrics['window_center_ms'] >= warmup_ms) &
                                    (windowed_metrics['window_center_ms'] < cooldown_start_ms)]
    cooldown_data = windowed_metrics[windowed_metrics['window_center_ms'] >= cooldown_start_ms]

    if len(active_data) == 0:
        print("Warning: Not enough data for convergence analysis")
        return

    print("\n" + "="*90)
    print("TRANSIENT PERIOD ANALYSIS")
    print("="*90)

    print(f"\nWarmup Period:   0 - {warmup_ms/1000:.0f}s ({warmup_ms/60000:.1f} min)")
    print(f"Active Period:   {warmup_ms/1000:.0f}s - {cooldown_start_ms/1000:.0f}s ({(cooldown_start_ms-warmup_ms)/60000:.1f} min)")
    print(f"Cooldown Period: {cooldown_start_ms/1000:.0f}s - {windowed_metrics['window_end_ms'].max()/1000:.0f}s")

    print(f"\nNumber of windows:")
    print(f"  Warmup:   {len(warmup_data)}")
    print(f"  Active:   {len(active_data)}")
    print(f"  Cooldown: {len(cooldown_data)}")

    # Compare metrics
    metrics_to_compare = [
        ('throughput', 'Throughput (commits/sec)', 1),
        ('success_rate', 'Success Rate (%)', 1),
        ('mean_latency', 'Mean Latency (ms)', 0),
        ('p95_latency', 'P95 Latency (ms)', 0),
    ]

    print(f"\n{'Metric':<25} {'Warmup':<15} {'Active':<15} {'Cooldown':<15} {'W→A Change':<15} {'C→A Change':<15}")
    print("-" * 100)

    for metric, label, decimals in metrics_to_compare:
        warmup_mean = warmup_data[metric].mean() if len(warmup_data) > 0 else 0
        active_mean = active_data[metric].mean()
        cooldown_mean = cooldown_data[metric].mean() if len(cooldown_data) > 0 else 0

        # Calculate changes
        if warmup_mean != 0:
            warmup_change_pct = ((active_mean - warmup_mean) / warmup_mean) * 100
            warmup_change_str = f"{warmup_change_pct:+.1f}%"
        else:
            warmup_change_str = "N/A"

        if cooldown_mean != 0 and active_mean != 0:
            cooldown_change_pct = ((cooldown_mean - active_mean) / active_mean) * 100
            cooldown_change_str = f"{cooldown_change_pct:+.1f}%"
        else:
            cooldown_change_str = "N/A"

        print(f"{label:<25} {warmup_mean:>10.{decimals}f}   {active_mean:>10.{decimals}f}   "
              f"{cooldown_mean:>10.{decimals}f}   {warmup_change_str:>12}   {cooldown_change_str:>12}")

    # Coefficient of variation (stability measure)
    print(f"\nCoefficient of Variation (lower = more stable):")
    print(f"{'Metric':<25} {'Warmup CV':<15} {'Active CV':<15} {'Cooldown CV':<15}")
    print("-" * 70)

    for metric, label, _ in metrics_to_compare:
        if active_data[metric].mean() > 0:
            warmup_cv = (warmup_data[metric].std() / warmup_data[metric].mean()) * 100 if len(warmup_data) > 0 and warmup_data[metric].mean() > 0 else 0
            active_cv = (active_data[metric].std() / active_data[metric].mean()) * 100
            cooldown_cv = (cooldown_data[metric].std() / cooldown_data[metric].mean()) * 100 if len(cooldown_data) > 0 and cooldown_data[metric].mean() > 0 else 0
            print(f"{label:<25} {warmup_cv:>10.1f}%      {active_cv:>10.1f}%      {cooldown_cv:>10.1f}%")

    print("\n" + "="*90)


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

    # Compute warmup and cooldown durations
    warmup_ms = compute_warmup_duration(config)
    cooldown_ms = warmup_ms  # Same formula for symmetry
    sim_duration_ms = config.get('simulation', {}).get('duration_ms', 3600000)
    active_window_ms = sim_duration_ms - warmup_ms - cooldown_ms

    print(f"\nComputed transient periods:")
    print(f"  Warmup:   {warmup_ms/1000:.0f}s ({warmup_ms/60000:.1f} min)")
    print(f"  Cooldown: {cooldown_ms/1000:.0f}s ({cooldown_ms/60000:.1f} min)")
    print(f"  Active:   {active_window_ms/1000:.0f}s ({active_window_ms/60000:.1f} min)")

    # Compute windowed metrics
    window_size_ms = args.window_size * 1000
    print(f"\nComputing metrics in {args.window_size}s windows...")
    windowed_metrics = compute_windowed_metrics(df, window_size_ms)
    print(f"Created {len(windowed_metrics)} windows")

    # Print convergence analysis
    cooldown_start_ms = sim_duration_ms - cooldown_ms
    print_convergence_analysis(windowed_metrics, warmup_ms, cooldown_start_ms)

    # Plot convergence
    if args.output:
        output_path = args.output
    else:
        output_path = f"warmup_validation_{exp_dir.parent.name}_{exp_dir.name}.png"

    plot_convergence(windowed_metrics, warmup_ms, cooldown_ms, sim_duration_ms, config, output_path)

    print(f"\nValidation complete!")


if __name__ == "__main__":
    main()
