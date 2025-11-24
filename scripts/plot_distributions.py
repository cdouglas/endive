#!/usr/bin/env python
"""
Plot theoretical distributions from experiment configurations.

This script visualizes the configured distributions (transaction runtime,
inter-arrival times, etc.) to understand what the simulator is configured to do,
independent of actual experimental results.
"""

import argparse
import os
from glob import glob
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import tomli


def load_experiment_config(exp_dir: str) -> Dict:
    """Load configuration from experiment directory."""
    cfg_path = os.path.join(exp_dir, "cfg.toml")
    if not os.path.exists(cfg_path):
        return None

    with open(cfg_path, 'rb') as f:
        return tomli.load(f)


def generate_lognormal_samples(mean_ms: float, sigma: float, min_ms: float, n_samples: int = 100000) -> np.ndarray:
    """
    Generate samples from lognormal distribution matching simulator implementation.

    The simulator uses:
        mu = log(mean_ms) - sigma^2/2
        sample = min_ms + lognormal(mu, sigma)

    So the total mean is min_ms + mean_ms (not just mean_ms!).
    """
    import math
    mu = math.log(mean_ms) - (sigma ** 2 / 2.0)
    samples = min_ms + np.random.lognormal(mu, sigma, n_samples)
    return samples


def generate_exponential_samples(scale: float, n_samples: int = 100000) -> np.ndarray:
    """Generate samples from exponential distribution."""
    return np.random.exponential(scale, n_samples)


def generate_transaction_runtime_table(configs: List[Dict], output_path: str):
    """Generate markdown table for transaction runtime distributions."""
    # Collect unique runtime configurations
    runtime_configs = {}
    for config in configs:
        if 'transaction' not in config or 'runtime' not in config['transaction']:
            continue

        runtime = config['transaction']['runtime']
        mean = runtime.get('mean', 10000)
        sigma = runtime.get('sigma', 1.5)
        min_val = runtime.get('min', 1000)

        key = (mean, sigma, min_val)
        label = config.get('experiment', {}).get('label', 'unknown')

        if key not in runtime_configs:
            runtime_configs[key] = label

    # Generate statistics for each configuration
    table_rows = []
    for (mean, sigma, min_val), label in runtime_configs.items():
        samples = generate_lognormal_samples(mean, sigma, min_val) / 1000.0  # Convert to seconds

        total_mean = (min_val + mean) / 1000.0
        observed_mean = np.mean(samples)
        p50 = np.percentile(samples, 50)
        p95 = np.percentile(samples, 95)
        p99 = np.percentile(samples, 99)

        table_rows.append({
            'label': label[:30],
            'min_s': min_val / 1000.0,
            'mean_lognorm_s': mean / 1000.0,
            'sigma': sigma,
            'total_mean_s': total_mean,
            'observed_mean_s': observed_mean,
            'p50_s': p50,
            'p95_s': p95,
            'p99_s': p99
        })

    # Write markdown table
    with open(output_path, 'w') as f:
        f.write("# Transaction Runtime Distribution\n\n")
        f.write("Theoretical distribution based on configuration parameters.\n\n")
        f.write("**Formula**: `runtime = MIN + lognormal(μ, σ)` where `μ = log(mean) - σ²/2`\n\n")
        f.write("**Total Expected Mean**: `MIN + mean` (not just `mean`)\n\n")
        f.write("## Configuration Parameters\n\n")
        f.write("| Configuration | MIN (s) | Mean Lognorm (s) | Sigma | Total Mean (s) |\n")
        f.write("|---------------|---------|------------------|-------|----------------|\n")
        for row in table_rows:
            f.write(f"| {row['label']} | {row['min_s']:.0f} | {row['mean_lognorm_s']:.0f} | {row['sigma']:.2f} | {row['total_mean_s']:.0f} |\n")

        f.write("\n## Theoretical Statistics (100k samples)\n\n")
        f.write("| Configuration | Observed Mean (s) | P50 (s) | P95 (s) | P99 (s) |\n")
        f.write("|---------------|-------------------|---------|---------|----------|\n")
        for row in table_rows:
            f.write(f"| {row['label']} | {row['observed_mean_s']:.0f} | {row['p50_s']:.0f} | {row['p95_s']:.0f} | {row['p99_s']:.0f} |\n")

    print(f"Saved transaction runtime table to {output_path}")


def plot_transaction_runtime(configs: List[Dict], output_path: str):
    """Plot transaction runtime distributions for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect unique runtime configurations
    runtime_configs = {}
    for config in configs:
        if 'transaction' not in config or 'runtime' not in config['transaction']:
            continue

        runtime = config['transaction']['runtime']
        mean = runtime.get('mean', 10000)
        sigma = runtime.get('sigma', 1.5)
        min_val = runtime.get('min', 1000)

        key = (mean, sigma, min_val)
        label = config.get('experiment', {}).get('label', 'unknown')

        if key not in runtime_configs:
            runtime_configs[key] = label

    # Plot each unique configuration
    for (mean, sigma, min_val), label in runtime_configs.items():
        samples = generate_lognormal_samples(mean, sigma, min_val)

        # Convert to seconds for readability
        samples_sec = samples / 1000.0

        # Plot PDF (histogram)
        total_mean_sec = (min_val + mean) / 1000.0
        ax1.hist(samples_sec, bins=100, alpha=0.5, density=True,
                label=f'{label[:20]}\n(total_mean={total_mean_sec:.0f}s: {min_val/1000:.0f}s + lognorm({mean/1000:.0f}s, {sigma}))')

        # Plot CDF
        sorted_samples = np.sort(samples_sec)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax2.plot(sorted_samples, cdf, linewidth=2, alpha=0.7,
                label=f'{label[:20]}\n(mean={np.mean(samples_sec):.0f}s, p50={np.percentile(samples_sec, 50):.0f}s)')

    ax1.set_xlabel('Transaction Runtime (seconds)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Transaction Runtime Distribution (PDF)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Transaction Runtime (seconds)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Transaction Runtime Distribution (CDF)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add percentile markers on CDF
    for p in [50, 95, 99]:
        ax2.axhline(p/100, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax2.text(ax2.get_xlim()[1] * 0.95, p/100, f'P{p}',
                ha='right', va='bottom', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved transaction runtime distribution to {output_path}")
    plt.close()


def generate_inter_arrival_table(configs: List[Dict], output_path: str):
    """Generate markdown table for inter-arrival time distributions."""
    # Collect unique inter-arrival configurations
    arrival_configs = {}
    for config in configs:
        if 'transaction' not in config or 'inter_arrival' not in config['transaction']:
            continue

        inter_arrival = config['transaction']['inter_arrival']
        distribution = inter_arrival.get('distribution', 'exponential')

        if distribution == 'exponential':
            scale = inter_arrival.get('scale', 500)
            if scale not in arrival_configs:
                arrival_configs[scale] = []

    # Generate statistics for each configuration
    table_rows = []
    for scale in sorted(arrival_configs.keys()):
        samples = generate_exponential_samples(scale)

        # Convert to appropriate units
        if scale >= 1000:
            samples_display = samples / 1000.0
            unit = 's'
        else:
            samples_display = samples
            unit = 'ms'

        mean_val = np.mean(samples_display)
        median_val = np.median(samples_display)
        p95 = np.percentile(samples_display, 95)
        p99 = np.percentile(samples_display, 99)

        # Arrival rate (transactions per second)
        arrival_rate = 1000.0 / scale

        table_rows.append({
            'scale_ms': scale,
            'unit': unit,
            'mean': mean_val,
            'median': median_val,
            'p95': p95,
            'p99': p99,
            'arrival_rate': arrival_rate
        })

    # Write markdown table
    with open(output_path, 'w') as f:
        f.write("# Inter-Arrival Time Distribution\n\n")
        f.write("Theoretical distribution based on configuration parameters.\n\n")
        f.write("**Formula**: `exponential(scale)`\n\n")
        f.write("**Properties**: mean = scale, median = scale × ln(2) ≈ 0.693 × scale\n\n")
        f.write("## Configuration Parameters\n\n")
        f.write("| Scale (ms) | Arrival Rate (txn/s) | Mean | Median | P95 | P99 |\n")
        f.write("|------------|----------------------|------|--------|-----|-----|\n")
        for row in table_rows:
            unit = row['unit']
            f.write(f"| {row['scale_ms']} | {row['arrival_rate']:.2f} | "
                   f"{row['mean']:.1f}{unit} | {row['median']:.1f}{unit} | "
                   f"{row['p95']:.1f}{unit} | {row['p99']:.1f}{unit} |\n")

        f.write("\n## Load Characterization\n\n")
        f.write("| Scale (ms) | Arrival Rate | Load Level | Expected Concurrency* |\n")
        f.write("|------------|--------------|------------|----------------------|\n")
        for row in table_rows:
            rate = row['arrival_rate']
            if rate > 50:
                load = "Very High"
            elif rate > 20:
                load = "High"
            elif rate > 5:
                load = "Medium"
            elif rate > 1:
                load = "Low"
            else:
                load = "Very Low"

            # Expected concurrency using Little's Law (assuming 180s mean runtime)
            mean_runtime_s = 180
            expected_concurrency = rate * mean_runtime_s

            f.write(f"| {row['scale_ms']} | {rate:.2f} txn/s | {load} | {expected_concurrency:.0f} |\n")

        f.write("\n*Expected concurrency assumes mean transaction runtime = 180s (baseline config)\n")
        f.write("\nLittle's Law: L = λ × W (concurrency = arrival_rate × mean_service_time)\n")

    print(f"Saved inter-arrival time table to {output_path}")


def plot_inter_arrival_times(configs: List[Dict], output_path: str):
    """Plot inter-arrival time distributions for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect unique inter-arrival configurations
    arrival_configs = {}
    for config in configs:
        if 'transaction' not in config or 'inter_arrival' not in config['transaction']:
            continue

        inter_arrival = config['transaction']['inter_arrival']
        distribution = inter_arrival.get('distribution', 'exponential')

        if distribution == 'exponential':
            scale = inter_arrival.get('scale', 500)
            label = config.get('experiment', {}).get('label', 'unknown')

            if scale not in arrival_configs:
                arrival_configs[scale] = []
            arrival_configs[scale].append(label)

    # Plot each unique configuration
    colors = plt.cm.viridis(np.linspace(0, 1, len(arrival_configs)))

    for idx, (scale, labels) in enumerate(sorted(arrival_configs.items())):
        samples = generate_exponential_samples(scale)

        # Convert to seconds for readability if scale > 1000
        if scale >= 1000:
            samples_display = samples / 1000.0
            unit = 's'
        else:
            samples_display = samples
            unit = 'ms'

        label_text = f'scale={scale}ms\n(mean={scale}{unit}, λ={1000/scale:.2f}/s)'

        # Plot PDF (histogram)
        ax1.hist(samples_display, bins=100, alpha=0.5, density=True,
                color=colors[idx], label=label_text)

        # Plot CDF
        sorted_samples = np.sort(samples_display)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax2.plot(sorted_samples, cdf, linewidth=2, alpha=0.7,
                color=colors[idx], label=label_text)

    unit_label = 'seconds' if any(s >= 1000 for s in arrival_configs.keys()) else 'milliseconds'

    ax1.set_xlabel(f'Inter-Arrival Time ({unit_label})', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
    ax1.set_title('Inter-Arrival Time Distribution (PDF)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    ax2.set_xlabel(f'Inter-Arrival Time ({unit_label})', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Inter-Arrival Time Distribution (CDF)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved inter-arrival time distribution to {output_path}")
    plt.close()


def generate_overview_table(configs: List[Dict], output_path: str):
    """Generate markdown table with combined distribution overview."""
    # Collect runtime configs
    runtime_configs = {}
    for config in configs:
        if 'transaction' not in config or 'runtime' not in config['transaction']:
            continue
        runtime = config['transaction']['runtime']
        key = (runtime.get('mean', 10000), runtime.get('sigma', 1.5), runtime.get('min', 1000))
        if key not in runtime_configs:
            runtime_configs[key] = config.get('experiment', {}).get('label', 'unknown')

    # Collect inter-arrival configs
    arrival_configs = {}
    for config in configs:
        if 'transaction' not in config or 'inter_arrival' not in config['transaction']:
            continue
        inter_arrival = config['transaction']['inter_arrival']
        scale = inter_arrival.get('scale', 500)
        if scale not in arrival_configs:
            arrival_configs[scale] = []

    # Write markdown
    with open(output_path, 'w') as f:
        f.write("# Experiment Distribution Overview\n\n")
        f.write("Summary of configured distributions across all experiments.\n\n")

        # Runtime configuration
        f.write("## Transaction Runtime Configuration\n\n")
        if runtime_configs:
            (mean, sigma, min_val) = list(runtime_configs.keys())[0]
            total_mean = (min_val + mean) / 1000.0
            f.write(f"**Distribution**: MIN + lognormal(μ, σ)\n\n")
            f.write(f"- **MIN**: {min_val/1000:.0f}s\n")
            f.write(f"- **Mean (lognormal part)**: {mean/1000:.0f}s\n")
            f.write(f"- **Sigma**: {sigma}\n")
            f.write(f"- **Total Expected Mean**: {total_mean:.0f}s\n\n")

        # Inter-arrival summary
        f.write("## Inter-Arrival Time Configurations\n\n")
        f.write(f"**Distribution**: exponential(scale)\n\n")
        f.write(f"**Number of load levels**: {len(arrival_configs)}\n\n")

        f.write("| Scale (ms) | Arrival Rate (txn/s) | Expected Concurrency* | Load Level |\n")
        f.write("|------------|----------------------|-----------------------|------------|\n")

        mean_runtime_s = 180
        for scale in sorted(arrival_configs.keys()):
            rate = 1000.0 / scale
            expected_concurrency = rate * mean_runtime_s

            if rate > 50:
                load = "Very High"
            elif rate > 20:
                load = "High"
            elif rate > 5:
                load = "Medium"
            elif rate > 1:
                load = "Low"
            else:
                load = "Very Low"

            f.write(f"| {scale} | {rate:.2f} | {expected_concurrency:.0f} | {load} |\n")

        f.write("\n*Expected concurrency using Little's Law: L = λ × W\n")
        f.write(f"\nAssuming mean transaction runtime = {mean_runtime_s}s (baseline configuration)\n\n")

        # Saturation analysis
        f.write("## Load Characterization\n\n")
        f.write("**Undersaturated** (L < 1): System has idle capacity, transactions complete immediately\n\n")
        f.write("**Light load** (1 ≤ L < 10): Low contention, high success rate (>99%)\n\n")
        f.write("**Medium load** (10 ≤ L < 100): Moderate contention, success rate 95-99%\n\n")
        f.write("**High load** (100 ≤ L < 1000): High contention, success rate 50-95%\n\n")
        f.write("**Very high load** (L ≥ 1000): Saturation, success rate <50%, exponential tail latency\n\n")

        # Experiment counts
        f.write("## Experiment Summary\n\n")
        f.write(f"- **Total configurations**: {len(configs)}\n")
        f.write(f"- **Runtime configurations**: {len(runtime_configs)}\n")
        f.write(f"- **Load levels**: {len(arrival_configs)}\n")

    print(f"Saved overview table to {output_path}")


def plot_combined_overview(configs: List[Dict], output_path: str):
    """Plot overview combining key distributions."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Transaction runtime CDF
    ax1 = fig.add_subplot(gs[0, :2])
    runtime_configs = {}
    for config in configs:
        if 'transaction' not in config or 'runtime' not in config['transaction']:
            continue
        runtime = config['transaction']['runtime']
        mean = runtime.get('mean', 10000)
        sigma = runtime.get('sigma', 1.5)
        min_val = runtime.get('min', 1000)
        key = (mean, sigma, min_val)
        label = config.get('experiment', {}).get('label', 'unknown')
        if key not in runtime_configs:
            runtime_configs[key] = label

    for (mean, sigma, min_val), label in runtime_configs.items():
        samples = generate_lognormal_samples(mean, sigma, min_val) / 1000.0
        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax1.plot(sorted_samples, cdf, linewidth=2, alpha=0.7,
                label=f'Runtime: μ={mean/1000:.0f}s, σ={sigma}')

    ax1.set_xlabel('Transaction Runtime (seconds)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Cumulative Probability', fontsize=11, fontweight='bold')
    ax1.set_title('Transaction Runtime (CDF)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Inter-arrival CDF (log scale for readability)
    ax2 = fig.add_subplot(gs[0, 2])
    arrival_configs = {}
    for config in configs:
        if 'transaction' not in config or 'inter_arrival' not in config['transaction']:
            continue
        inter_arrival = config['transaction']['inter_arrival']
        scale = inter_arrival.get('scale', 500)
        if scale not in arrival_configs:
            arrival_configs[scale] = []

    for scale in sorted(arrival_configs.keys())[:8]:  # Limit to 8 for readability
        samples = generate_exponential_samples(scale)
        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        ax2.plot(sorted_samples, cdf, linewidth=2, alpha=0.7,
                label=f'{scale}ms')

    ax2.set_xlabel('Inter-Arrival (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('CDF', fontsize=11, fontweight='bold')
    ax2.set_title('Inter-Arrival Times', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='lower right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3, which='both')

    # Arrival rate vs expected concurrency
    ax3 = fig.add_subplot(gs[1, :])

    # For each inter-arrival scale, compute arrival rate and expected concurrency
    arrival_rates = []
    expected_concurrency = []
    labels = []

    for scale in sorted(arrival_configs.keys()):
        # Arrival rate (transactions per second)
        arrival_rate = 1000.0 / scale  # scale is in ms

        # Expected concurrency using Little's Law: L = λ × W
        # λ = arrival rate, W = mean service time (transaction runtime)
        # Use the first runtime config (should be same for all baseline experiments)
        if runtime_configs:
            (mean_runtime, _, _) = list(runtime_configs.keys())[0]
            mean_runtime_sec = mean_runtime / 1000.0
            exp_concurrency = arrival_rate * mean_runtime_sec
        else:
            exp_concurrency = 0

        arrival_rates.append(arrival_rate)
        expected_concurrency.append(exp_concurrency)
        labels.append(f'{scale}ms')

    bars = ax3.bar(range(len(arrival_rates)), expected_concurrency, alpha=0.7, color='steelblue')
    ax3.set_xlabel('Inter-Arrival Time', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Expected Concurrency\n(Little\'s Law: L = λ × W)', fontsize=11, fontweight='bold')
    ax3.set_title('Expected Concurrent Transactions by Load Level', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, rate, conc) in enumerate(zip(bars, arrival_rates, expected_concurrency)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{conc:.1f}\n({rate:.2f} txn/s)',
                ha='center', va='bottom', fontsize=8)

    # Add horizontal line at concurrency = 1 (undersaturated)
    ax3.axhline(1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax3.text(len(arrival_rates) - 1, 1.1, 'Undersaturated (L < 1)',
            ha='right', va='bottom', fontsize=9, color='red')

    plt.suptitle('Experiment Distribution Overview', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined overview to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot theoretical distributions from experiment configurations"
    )
    parser.add_argument(
        "-i", "--input-dir",
        default="experiments",
        help="Base directory containing experiment results (default: experiments)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="plots/distributions",
        help="Output directory for plots (default: plots/distributions)"
    )
    parser.add_argument(
        "-p", "--pattern",
        default="exp2_*",
        help="Pattern to match experiment directories (default: exp2_*)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Scan for experiment directories
    search_pattern = os.path.join(args.input_dir, args.pattern)
    exp_dirs = glob(search_pattern)

    if not exp_dirs:
        print(f"No experiments found matching {search_pattern}")
        return

    print(f"Found {len(exp_dirs)} experiment directories")

    # Load all configurations
    configs = []
    for exp_dir in exp_dirs:
        config = load_experiment_config(exp_dir)
        if config:
            configs.append(config)

    if not configs:
        print("No valid configurations found")
        return

    print(f"Loaded {len(configs)} configurations")

    # Generate plots and tables
    print("\nGenerating distribution plots and tables...")

    # Transaction runtime
    plot_transaction_runtime(
        configs,
        os.path.join(args.output_dir, "transaction_runtime.png")
    )
    generate_transaction_runtime_table(
        configs,
        os.path.join(args.output_dir, "transaction_runtime.md")
    )

    # Inter-arrival times
    plot_inter_arrival_times(
        configs,
        os.path.join(args.output_dir, "inter_arrival_times.png")
    )
    generate_inter_arrival_table(
        configs,
        os.path.join(args.output_dir, "inter_arrival_times.md")
    )

    # Overview
    plot_combined_overview(
        configs,
        os.path.join(args.output_dir, "overview.png")
    )
    generate_overview_table(
        configs,
        os.path.join(args.output_dir, "overview.md")
    )

    print("\nAll plots and tables generated successfully!")


if __name__ == "__main__":
    main()
