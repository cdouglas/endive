#!/usr/bin/env python3
"""
Generate parameter sensitivity plots for experiments that vary parameters
at a fixed load level (not saturation sweeps).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_multi_table_mixed():
    """Generate heatmap for multi_table_mixed experiment."""
    df = pd.read_csv('plots/multi_table_mixed/experiment_index.csv')

    # Create pivot tables for throughput and latency
    pivot_throughput = df.pivot_table(
        index='num_tables',
        columns='real_conflict_probability',
        values='throughput',
        aggfunc='first'
    )

    pivot_latency = df.pivot_table(
        index='num_tables',
        columns='real_conflict_probability',
        values='p50_commit_latency',
        aggfunc='first'
    )

    pivot_success = df.pivot_table(
        index='num_tables',
        columns='real_conflict_probability',
        values='success_rate',
        aggfunc='first'
    )

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Throughput heatmap
    im1 = axes[0].imshow(pivot_throughput.values, cmap='RdYlGn', aspect='auto')
    axes[0].set_xticks(range(len(pivot_throughput.columns)))
    axes[0].set_xticklabels([f'{x:.1f}' for x in pivot_throughput.columns])
    axes[0].set_yticks(range(len(pivot_throughput.index)))
    axes[0].set_yticklabels(pivot_throughput.index)
    axes[0].set_xlabel('Real Conflict Probability')
    axes[0].set_ylabel('Number of Tables')
    axes[0].set_title('Throughput (commits/s)')
    plt.colorbar(im1, ax=axes[0])

    # Add text annotations
    for i in range(len(pivot_throughput.index)):
        for j in range(len(pivot_throughput.columns)):
            val = pivot_throughput.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9)

    # Latency heatmap (inverse colormap - lower is better)
    im2 = axes[1].imshow(pivot_latency.values / 1000, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_xticks(range(len(pivot_latency.columns)))
    axes[1].set_xticklabels([f'{x:.1f}' for x in pivot_latency.columns])
    axes[1].set_yticks(range(len(pivot_latency.index)))
    axes[1].set_yticklabels(pivot_latency.index)
    axes[1].set_xlabel('Real Conflict Probability')
    axes[1].set_ylabel('Number of Tables')
    axes[1].set_title('P50 Commit Latency (s)')
    plt.colorbar(im2, ax=axes[1])

    for i in range(len(pivot_latency.index)):
        for j in range(len(pivot_latency.columns)):
            val = pivot_latency.values[i, j] / 1000
            if not np.isnan(val):
                axes[1].text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=9)

    # Success rate heatmap
    im3 = axes[2].imshow(pivot_success.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    axes[2].set_xticks(range(len(pivot_success.columns)))
    axes[2].set_xticklabels([f'{x:.1f}' for x in pivot_success.columns])
    axes[2].set_yticks(range(len(pivot_success.index)))
    axes[2].set_yticklabels(pivot_success.index)
    axes[2].set_xlabel('Real Conflict Probability')
    axes[2].set_ylabel('Number of Tables')
    axes[2].set_title('Success Rate (%)')
    plt.colorbar(im3, ax=axes[2])

    for i in range(len(pivot_success.index)):
        for j in range(len(pivot_success.columns)):
            val = pivot_success.values[i, j]
            if not np.isnan(val):
                axes[2].text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9)

    plt.suptitle('Multi-Table Mixed: Parameter Sensitivity at inter_arrival_scale=100', fontsize=14)
    plt.tight_layout()

    output_path = Path('plots/multi_table_mixed/parameter_sensitivity.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")

    # Also generate markdown table
    def df_to_md(df):
        """Convert DataFrame to markdown table without tabulate."""
        lines = []
        # Header
        lines.append("| Tables | " + " | ".join(f"{c:.1f}" for c in df.columns) + " |")
        lines.append("|--------|" + "|".join(["------"] * len(df.columns)) + "|")
        # Data rows
        for idx, row in df.iterrows():
            vals = " | ".join(f"{v:.2f}" if not pd.isna(v) else "-" for v in row.values)
            lines.append(f"| {idx} | {vals} |")
        return "\n".join(lines)

    md_lines = [
        "# Multi-Table Mixed: Parameter Sensitivity",
        "",
        "All experiments run at inter_arrival_scale = 100 (moderate load).",
        "",
        "## Throughput (commits/s)",
        "",
        df_to_md(pivot_throughput),
        "",
        "## P50 Commit Latency (seconds)",
        "",
        df_to_md(pivot_latency / 1000),
        "",
        "## Success Rate (%)",
        "",
        df_to_md(pivot_success),
    ]

    md_path = Path('plots/multi_table_mixed/parameter_sensitivity.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")

if __name__ == '__main__':
    plot_multi_table_mixed()
