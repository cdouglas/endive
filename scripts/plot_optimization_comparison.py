#!/usr/bin/env python3
"""
Generate comparison plots for optimization experiments.

Compares baseline, ml_append, metadata, and combined_optimizations
for each storage provider (azure, azurex, s3, s3x).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_experiment_index(exp_dir: Path) -> pd.DataFrame | None:
    """Load experiment_index.csv from an experiment directory."""
    index_path = exp_dir / "experiment_index.csv"
    if not index_path.exists():
        return None
    return pd.read_csv(index_path)

def plot_latency_comparison(provider: str, output_dir: Path):
    """Generate latency vs throughput comparison plot for a provider."""
    experiments = {
        'baseline': f'plots/baseline_{provider}',
        'metadata': f'plots/metadata_{provider}',
        'ml_append': f'plots/ml_append_{provider}',
        'combined': f'plots/combined_optimizations_{provider}',
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'baseline': '#1f77b4',
        'metadata': '#ff7f0e',
        'ml_append': '#2ca02c',
        'combined': '#d62728',
    }

    labels = {
        'baseline': 'Baseline',
        'metadata': 'Metadata Inlining',
        'ml_append': 'ML+ (Append)',
        'combined': 'Combined (Metadata + ML+)',
    }

    markers = {
        'baseline': 'o',
        'metadata': 's',
        'ml_append': '^',
        'combined': 'D',
    }

    data_found = False

    for exp_name, exp_path in experiments.items():
        df = load_experiment_index(Path(exp_path))
        if df is None:
            print(f"  Skipping {exp_name}: no data found at {exp_path}")
            continue

        data_found = True

        # Sort by throughput
        df = df.sort_values('throughput')

        # Plot P50 latency vs throughput
        ax.plot(df['throughput'], df['p50_commit_latency'] / 1000,  # Convert to seconds
                marker=markers[exp_name], color=colors[exp_name],
                label=labels[exp_name], linewidth=2, markersize=8)

    if not data_found:
        print(f"  No data found for provider {provider}")
        plt.close(fig)
        return False

    ax.set_xlabel('Throughput (commits/s)', fontsize=12)
    ax.set_ylabel('P50 Commit Latency (s)', fontsize=12)
    ax.set_title(f'Latency vs Throughput - {provider.upper()}', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'latency_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True

def plot_success_comparison(provider: str, output_dir: Path):
    """Generate success rate vs throughput comparison plot."""
    experiments = {
        'baseline': f'plots/baseline_{provider}',
        'metadata': f'plots/metadata_{provider}',
        'ml_append': f'plots/ml_append_{provider}',
        'combined': f'plots/combined_optimizations_{provider}',
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        'baseline': '#1f77b4',
        'metadata': '#ff7f0e',
        'ml_append': '#2ca02c',
        'combined': '#d62728',
    }

    labels = {
        'baseline': 'Baseline',
        'metadata': 'Metadata Inlining',
        'ml_append': 'ML+ (Append)',
        'combined': 'Combined (Metadata + ML+)',
    }

    markers = {
        'baseline': 'o',
        'metadata': 's',
        'ml_append': '^',
        'combined': 'D',
    }

    data_found = False

    for exp_name, exp_path in experiments.items():
        df = load_experiment_index(Path(exp_path))
        if df is None:
            continue

        data_found = True
        df = df.sort_values('throughput')

        ax.plot(df['throughput'], df['success_rate'],
                marker=markers[exp_name], color=colors[exp_name],
                label=labels[exp_name], linewidth=2, markersize=8)

    if not data_found:
        plt.close(fig)
        return False

    ax.set_xlabel('Throughput (commits/s)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Success Rate vs Throughput - {provider.upper()}', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    # Add 99% threshold line
    ax.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')

    output_path = output_dir / 'success_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return True

def generate_markdown_table(provider: str, output_dir: Path):
    """Generate markdown comparison table."""
    experiments = {
        'baseline': f'plots/baseline_{provider}',
        'metadata': f'plots/metadata_{provider}',
        'ml_append': f'plots/ml_append_{provider}',
        'combined': f'plots/combined_optimizations_{provider}',
    }

    labels = {
        'baseline': 'Baseline',
        'metadata': 'Metadata Inlining',
        'ml_append': 'ML+ (Append)',
        'combined': 'Combined',
    }

    all_data = []

    for exp_name, exp_path in experiments.items():
        df = load_experiment_index(Path(exp_path))
        if df is None:
            continue
        df['optimization'] = labels[exp_name]
        all_data.append(df)

    if not all_data:
        return False

    combined = pd.concat(all_data, ignore_index=True)

    # Find sustainable throughput (highest throughput with >= 99% success rate)
    sustainable = combined[combined['success_rate'] >= 99].groupby('optimization').agg({
        'throughput': 'max',
        'p50_commit_latency': 'min',
        'success_rate': 'max'
    }).reset_index()

    # Generate markdown
    md_lines = [
        f"# Optimization Comparison - {provider.upper()}",
        "",
        "## Sustainable Throughput (>= 99% success rate)",
        "",
        "| Optimization | Max Throughput (c/s) | Min P50 Latency (s) |",
        "|--------------|---------------------|---------------------|",
    ]

    for _, row in sustainable.sort_values('throughput', ascending=False).iterrows():
        md_lines.append(
            f"| {row['optimization']} | {row['throughput']:.2f} | {row['p50_commit_latency']/1000:.2f} |"
        )

    md_lines.extend(["", "## All Data Points", ""])

    # Pivot table for comparison
    pivot = combined.pivot_table(
        index='inter_arrival_scale',
        columns='optimization',
        values=['throughput', 'p50_commit_latency', 'success_rate'],
        aggfunc='first'
    )

    md_lines.append("### Throughput by Load Level")
    md_lines.append("")
    md_lines.append("| Scale | " + " | ".join(labels.values()) + " |")
    md_lines.append("|-------|" + "|".join(["-------"] * len(labels)) + "|")

    for scale in sorted(combined['inter_arrival_scale'].unique()):
        row = [f"{scale:.0f}"]
        for opt in labels.values():
            try:
                val = pivot.loc[scale, ('throughput', opt)]
                row.append(f"{val:.2f}")
            except KeyError:
                row.append("-")
        md_lines.append("| " + " | ".join(row) + " |")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'comparison.md'
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"  Saved: {output_path}")
    return True

def main():
    providers = ['azure', 'azurex', 's3', 's3x']

    for provider in providers:
        print(f"\nGenerating comparison for {provider}...")
        output_dir = Path(f'plots/optimization_comparison/{provider}')

        plot_latency_comparison(provider, output_dir)
        plot_success_comparison(provider, output_dir)
        generate_markdown_table(provider, output_dir)

if __name__ == '__main__':
    main()
