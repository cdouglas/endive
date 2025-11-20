#!/usr/bin/env python
"""
Analysis and plotting module for icecap experiment results.

This module provides functions to generate:
1. CDF plots for commit latency across different client loads
2. Success rate plots vs client load
3. Impact of catalog latency on commit latency
"""

import argparse
import os
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, labs, theme_minimal,
    theme, element_text, scale_color_discrete, facet_wrap
)


def load_experiment_results(pattern: str) -> List[Tuple[str, pd.DataFrame]]:
    """Load all Parquet files matching the pattern."""
    files = sorted(glob(pattern))
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")

    results = []
    for filepath in files:
        df = pd.read_parquet(filepath)
        results.append((filepath, df))

    return results


def extract_param_from_filename(filename: str, param: str) -> float:
    """Extract parameter value from filename."""
    basename = os.path.basename(filename)

    if param == "inter_arrival":
        match = re.search(r'ia_(\d+(?:\.\d+)?)', basename)
        if match:
            return float(match.group(1))
    elif param == "cas_latency":
        match = re.search(r'cas_(\d+)', basename)
        if match:
            return float(match.group(1))

    return None


def plot_commit_latency_cdf(
    results: List[Tuple[str, pd.DataFrame]],
    output_path: str,
    param: str = "inter_arrival"
):
    """
    Plot CDF of commit latency for different parameter values.

    Args:
        results: List of (filename, dataframe) tuples
        output_path: Path to save the plot
        param: Parameter to extract from filename ("inter_arrival" or "cas_latency")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort results by parameter value for consistent legend ordering
    results_with_param = []
    for filepath, df in results:
        param_value = extract_param_from_filename(filepath, param)
        if param_value is not None:
            results_with_param.append((param_value, filepath, df))

    results_with_param.sort(key=lambda x: x[0])

    for param_value, filepath, df in results_with_param:
        # Filter only committed transactions
        committed = df[df['status'] == 'committed']

        if len(committed) == 0:
            print(f"Warning: No committed transactions in {filepath}")
            continue

        # Calculate CDF
        latencies = np.sort(committed['commit_latency'].values)
        cdf = np.arange(1, len(latencies) + 1) / len(latencies)

        # Plot
        label_name = "Inter-arrival" if param == "inter_arrival" else "CAS latency"
        ax.plot(latencies, cdf, label=f'{label_name}={param_value}ms', linewidth=2)

    ax.set_xlabel('Commit Latency (ms)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('CDF of Commit Latency', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved CDF plot to {output_path}")
    plt.close()


def plot_success_rate(
    results: List[Tuple[str, pd.DataFrame]],
    output_path: str,
    param: str = "inter_arrival"
):
    """
    Plot success rate vs parameter value.

    Args:
        results: List of (filename, dataframe) tuples
        output_path: Path to save the plot
        param: Parameter to extract from filename ("inter_arrival" or "cas_latency")
    """
    data_points = []

    for filepath, df in results:
        param_value = extract_param_from_filename(filepath, param)
        if param_value is None:
            continue

        total = len(df)
        committed = len(df[df['status'] == 'committed'])
        success_rate = 100 * committed / total if total > 0 else 0

        # Calculate effective throughput (transactions per second)
        if len(df) > 0:
            duration_s = df['t_submit'].max() / 1000  # convert ms to seconds
            throughput = committed / duration_s if duration_s > 0 else 0
        else:
            throughput = 0

        data_points.append({
            'param_value': param_value,
            'success_rate': success_rate,
            'total_txns': total,
            'committed': committed,
            'aborted': total - committed,
            'throughput': throughput
        })

    if not data_points:
        print("Warning: No data points to plot")
        return

    df_plot = pd.DataFrame(data_points).sort_values('param_value')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Success rate
    ax1.plot(df_plot['param_value'], df_plot['success_rate'],
             marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Inter-arrival Time (ms)' if param == 'inter_arrival' else 'CAS Latency (ms)',
                   fontsize=12)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Transaction Success Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])

    # Add value labels
    for _, row in df_plot.iterrows():
        ax1.annotate(f"{row['success_rate']:.1f}%",
                    (row['param_value'], row['success_rate']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=9)

    # Plot 2: Throughput
    ax2.plot(df_plot['param_value'], df_plot['throughput'],
             marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Inter-arrival Time (ms)' if param == 'inter_arrival' else 'CAS Latency (ms)',
                   fontsize=12)
    ax2.set_ylabel('Throughput (commits/sec)', fontsize=12)
    ax2.set_title('Committed Transaction Throughput', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved success rate plot to {output_path}")
    plt.close()


def plot_catalog_latency_impact(
    results: List[Tuple[str, pd.DataFrame]],
    output_path: str
):
    """
    Plot the impact of catalog latency on commit latency.

    This expects results from a combined sweep (both inter-arrival and CAS latency).
    """
    data_points = []

    for filepath, df in results:
        ia_time = extract_param_from_filename(filepath, "inter_arrival")
        cas_latency = extract_param_from_filename(filepath, "cas_latency")

        if ia_time is None or cas_latency is None:
            continue

        committed = df[df['status'] == 'committed']
        if len(committed) == 0:
            continue

        data_points.append({
            'inter_arrival': ia_time,
            'cas_latency': cas_latency,
            'mean_commit_latency': committed['commit_latency'].mean(),
            'median_commit_latency': committed['commit_latency'].median(),
            'p95_commit_latency': committed['commit_latency'].quantile(0.95),
            'p99_commit_latency': committed['commit_latency'].quantile(0.99),
            'mean_retries': committed['n_retries'].mean(),
            'success_rate': 100 * len(committed) / len(df)
        })

    if not data_points:
        print("Warning: No data points to plot for catalog latency impact")
        return

    df_plot = pd.DataFrame(data_points)

    # Create a grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get unique inter-arrival times for different lines
    ia_times = sorted(df_plot['inter_arrival'].unique())

    # Plot 1: Mean commit latency vs CAS latency
    ax = axes[0, 0]
    for ia_time in ia_times:
        subset = df_plot[df_plot['inter_arrival'] == ia_time].sort_values('cas_latency')
        ax.plot(subset['cas_latency'], subset['mean_commit_latency'],
                marker='o', linewidth=2, label=f'IA={ia_time}ms')
    ax.set_xlabel('CAS Latency (ms)', fontsize=11)
    ax.set_ylabel('Mean Commit Latency (ms)', fontsize=11)
    ax.set_title('Mean Commit Latency vs CAS Latency', fontsize=12, fontweight='bold')
    ax.legend(title='Inter-arrival time', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: P95 commit latency vs CAS latency
    ax = axes[0, 1]
    for ia_time in ia_times:
        subset = df_plot[df_plot['inter_arrival'] == ia_time].sort_values('cas_latency')
        ax.plot(subset['cas_latency'], subset['p95_commit_latency'],
                marker='s', linewidth=2, label=f'IA={ia_time}ms')
    ax.set_xlabel('CAS Latency (ms)', fontsize=11)
    ax.set_ylabel('P95 Commit Latency (ms)', fontsize=11)
    ax.set_title('P95 Commit Latency vs CAS Latency', fontsize=12, fontweight='bold')
    ax.legend(title='Inter-arrival time', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Mean retries vs CAS latency
    ax = axes[1, 0]
    for ia_time in ia_times:
        subset = df_plot[df_plot['inter_arrival'] == ia_time].sort_values('cas_latency')
        ax.plot(subset['cas_latency'], subset['mean_retries'],
                marker='^', linewidth=2, label=f'IA={ia_time}ms')
    ax.set_xlabel('CAS Latency (ms)', fontsize=11)
    ax.set_ylabel('Mean Retries per Transaction', fontsize=11)
    ax.set_title('Retry Rate vs CAS Latency', fontsize=12, fontweight='bold')
    ax.legend(title='Inter-arrival time', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 4: Success rate vs CAS latency
    ax = axes[1, 1]
    for ia_time in ia_times:
        subset = df_plot[df_plot['inter_arrival'] == ia_time].sort_values('cas_latency')
        ax.plot(subset['cas_latency'], subset['success_rate'],
                marker='D', linewidth=2, label=f'IA={ia_time}ms')
    ax.set_xlabel('CAS Latency (ms)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Success Rate vs CAS Latency', fontsize=12, fontweight='bold')
    ax.legend(title='Inter-arrival time', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved catalog latency impact plot to {output_path}")
    plt.close()


def generate_summary_table(
    results: List[Tuple[str, pd.DataFrame]],
    output_path: str
):
    """Generate a summary table of all experiment results."""
    rows = []

    for filepath, df in results:
        ia_time = extract_param_from_filename(filepath, "inter_arrival")
        cas_latency = extract_param_from_filename(filepath, "cas_latency")

        committed = df[df['status'] == 'committed']
        total = len(df)

        if len(committed) > 0:
            rows.append({
                'file': os.path.basename(filepath),
                'inter_arrival_ms': ia_time if ia_time else '-',
                'cas_latency_ms': cas_latency if cas_latency else '-',
                'total_txns': total,
                'committed': len(committed),
                'aborted': total - len(committed),
                'success_rate_%': 100 * len(committed) / total,
                'mean_commit_latency_ms': committed['commit_latency'].mean(),
                'median_commit_latency_ms': committed['commit_latency'].median(),
                'p95_commit_latency_ms': committed['commit_latency'].quantile(0.95),
                'p99_commit_latency_ms': committed['commit_latency'].quantile(0.99),
                'mean_retries': committed['n_retries'].mean(),
                'max_retries': committed['n_retries'].max(),
            })

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_path, index=False, float_format='%.2f')
    print(f"Saved summary table to {output_path}")


def cli():
    parser = argparse.ArgumentParser(
        description="Analyze and plot icecap experiment results"
    )
    parser.add_argument(
        "-i", "--input-dir",
        default="experiments",
        help="Input directory containing Parquet files (default: experiments)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots",
        help="Output directory for plots (default: plots)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.parquet",
        help="File pattern to match (default: *.parquet)"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # CDF plot
    cdf_parser = subparsers.add_parser("cdf", help="Plot CDF of commit latency")
    cdf_parser.add_argument(
        "--param",
        choices=["inter_arrival", "cas_latency"],
        default="inter_arrival",
        help="Parameter to group by (default: inter_arrival)"
    )

    # Success rate plot
    success_parser = subparsers.add_parser(
        "success-rate",
        help="Plot success rate vs parameter"
    )
    success_parser.add_argument(
        "--param",
        choices=["inter_arrival", "cas_latency"],
        default="inter_arrival",
        help="Parameter to plot (default: inter_arrival)"
    )

    # Catalog latency impact
    subparsers.add_parser(
        "latency-impact",
        help="Plot catalog latency impact (requires combined sweep results)"
    )

    # Generate all plots
    subparsers.add_parser("all", help="Generate all plots and summary table")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    pattern = os.path.join(args.input_dir, args.pattern)
    print(f"Loading results from: {pattern}")
    results = load_experiment_results(pattern)
    print(f"Loaded {len(results)} result files")

    # Generate plots based on command
    if args.command == "cdf":
        output_path = os.path.join(args.output_dir, "cdf_commit_latency.png")
        plot_commit_latency_cdf(results, output_path, args.param)

    elif args.command == "success-rate":
        output_path = os.path.join(args.output_dir, "success_rate.png")
        plot_success_rate(results, output_path, args.param)

    elif args.command == "latency-impact":
        output_path = os.path.join(args.output_dir, "catalog_latency_impact.png")
        plot_catalog_latency_impact(results, output_path)

    elif args.command == "all":
        print("\nGenerating all plots...")

        # Try CDF plot with inter-arrival
        try:
            output_path = os.path.join(args.output_dir, "cdf_commit_latency_clients.png")
            plot_commit_latency_cdf(results, output_path, "inter_arrival")
        except Exception as e:
            print(f"Could not generate CDF (inter-arrival): {e}")

        # Try CDF plot with CAS latency
        try:
            output_path = os.path.join(args.output_dir, "cdf_commit_latency_cas.png")
            plot_commit_latency_cdf(results, output_path, "cas_latency")
        except Exception as e:
            print(f"Could not generate CDF (CAS latency): {e}")

        # Try success rate with inter-arrival
        try:
            output_path = os.path.join(args.output_dir, "success_rate_clients.png")
            plot_success_rate(results, output_path, "inter_arrival")
        except Exception as e:
            print(f"Could not generate success rate (inter-arrival): {e}")

        # Try success rate with CAS latency
        try:
            output_path = os.path.join(args.output_dir, "success_rate_cas.png")
            plot_success_rate(results, output_path, "cas_latency")
        except Exception as e:
            print(f"Could not generate success rate (CAS latency): {e}")

        # Try catalog latency impact (requires combined sweep)
        try:
            output_path = os.path.join(args.output_dir, "catalog_latency_impact.png")
            plot_catalog_latency_impact(results, output_path)
        except Exception as e:
            print(f"Could not generate latency impact plot: {e}")

        # Generate summary table
        try:
            output_path = os.path.join(args.output_dir, "summary.csv")
            generate_summary_table(results, output_path)
        except Exception as e:
            print(f"Could not generate summary table: {e}")

        print("\nAll plots generated!")


if __name__ == "__main__":
    cli()
