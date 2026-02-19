#!/usr/bin/env python3
"""Generate heatmaps for exp2_mix_heatmap 2D parameter sweep."""

import os
import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def extract_experiment_params(experiments_dir: Path) -> pd.DataFrame:
    """Extract parameters from all exp2_mix_heatmap experiments."""
    records = []

    for exp_dir in experiments_dir.glob("exp2_mix_heatmap-*"):
        cfg_path = exp_dir / "cfg.toml"
        if not cfg_path.exists():
            continue

        with open(cfg_path, "rb") as f:
            cfg = tomllib.load(f)

        txn = cfg.get("transaction", {})
        fa_ratio = txn.get("operation_types", {}).get("fast_append", 0.5)
        scale = txn.get("inter_arrival", {}).get("scale", 100.0)

        # Count seeds
        seed_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        num_seeds = len(seed_dirs)

        records.append({
            "hash": exp_dir.name.split("-")[-1],
            "exp_dir": str(exp_dir),
            "fa_ratio": fa_ratio,
            "inter_arrival_scale": scale,
            "num_seeds": num_seeds,
        })

    return pd.DataFrame(records)


def load_experiment_results(params_df: pd.DataFrame) -> pd.DataFrame:
    """Load results from experiment directories and aggregate by seed."""
    all_results = []

    for _, row in params_df.iterrows():
        exp_dir = Path(row["exp_dir"])
        seed_results = []

        for seed_dir in exp_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            results_path = seed_dir / "results.parquet"
            if not results_path.exists():
                continue

            df = pd.read_parquet(results_path)
            total = len(df)
            committed = (df["status"].str.lower() == "committed").sum()

            # Filter to steady state (middle 80% of simulation)
            duration = df["t_commit"].max() - df["t_submit"].min()
            warmup = duration * 0.1
            cooldown = duration * 0.1
            t_min = df["t_submit"].min() + warmup
            t_max = df["t_commit"].max() - cooldown
            df_steady = df[(df["t_submit"] >= t_min) & (df["t_commit"] <= t_max)]

            steady_committed = df_steady[df_steady["status"].str.lower() == "committed"]
            if len(steady_committed) == 0:
                continue

            # Compute metrics
            throughput = len(steady_committed) / (duration / 1000 / 3600)  # per hour
            latency = steady_committed["commit_latency"]  # Pre-computed in parquet

            seed_results.append({
                "total_txns": total,
                "committed": committed,
                "success_rate": committed / total * 100 if total > 0 else 0,
                "throughput": throughput,
                "mean_latency": latency.mean(),
                "p50_latency": latency.median(),
                "p95_latency": latency.quantile(0.95),
                "p99_latency": latency.quantile(0.99),
            })

        if seed_results:
            # Aggregate across seeds
            seed_df = pd.DataFrame(seed_results)
            all_results.append({
                "hash": row["hash"],
                "fa_ratio": row["fa_ratio"],
                "inter_arrival_scale": row["inter_arrival_scale"],
                "num_seeds": len(seed_results),
                "success_rate": seed_df["success_rate"].mean(),
                "success_rate_std": seed_df["success_rate"].std(),
                "throughput": seed_df["throughput"].mean(),
                "throughput_std": seed_df["throughput"].std(),
                "mean_latency": seed_df["mean_latency"].mean(),
                "p50_latency": seed_df["p50_latency"].mean(),
                "p95_latency": seed_df["p95_latency"].mean(),
                "p99_latency": seed_df["p99_latency"].mean(),
            })

    return pd.DataFrame(all_results)


def create_heatmap(data: pd.DataFrame, value_col: str, title: str, output_path: Path,
                   cmap: str = "viridis", vmin: float = None, vmax: float = None,
                   fmt: str = ".1f"):
    """Create a heatmap from the 2D parameter sweep data."""
    # Pivot to 2D grid
    pivot = data.pivot_table(
        index="fa_ratio",
        columns="inter_arrival_scale",
        values=value_col,
        aggfunc="mean"
    )

    # Sort index (fa_ratio) descending for visual clarity
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])

    ax.set_xlabel("Inter-arrival Scale (ms)")
    ax.set_ylabel("FastAppend Ratio")
    ax.set_title(title)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < (vmax or pivot.values.max()) * 0.5 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                       color=text_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def generate_vo_heatmaps(experiments_dir: Path, output_dir: Path):
    """Generate ValidatedOverwrite-specific heatmaps."""
    # Check if operation_type_metrics.csv exists
    metrics_path = output_dir / "operation_type_metrics.csv"
    if not metrics_path.exists():
        print("  Skipping VO heatmaps (run analyze_operation_types.py first)")
        return

    df = pd.read_csv(metrics_path)
    vo = df[df["operation_type"] == "validated_overwrite"].copy()

    if len(vo) == 0:
        print("  No ValidatedOverwrite data found")
        return

    # VO Success Rate
    pivot = vo.pivot_table(
        index="fa_ratio", columns="inter_arrival_scale",
        values="success_rate", aggfunc="mean"
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=50, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
    ax.set_xlabel("Inter-arrival Scale (ms)")
    ax.set_ylabel("FastAppend Ratio")
    ax.set_title("ValidatedOverwrite Success Rate (%) by Operation Mix and Load")
    fig.colorbar(im, ax=ax, label="Success Rate (%)")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val < 75 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                       color=color, fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_vo_success_rate.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'heatmap_vo_success_rate.png'}")

    # VO P95 Latency
    pivot = vo.pivot_table(
        index="fa_ratio", columns="inter_arrival_scale",
        values="p95_latency", aggfunc="mean"
    ).sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="inferno_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.0f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:.1f}" for r in pivot.index])
    ax.set_xlabel("Inter-arrival Scale (ms)")
    ax.set_ylabel("FastAppend Ratio")
    ax.set_title("ValidatedOverwrite P95 Latency (ms) by Operation Mix and Load")
    fig.colorbar(im, ax=ax, label="P95 Latency (ms)")

    vmax = np.nanmax(pivot.values)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > vmax * 0.4 else "black"
                label = f"{val/1000:.1f}s" if val >= 1000 else f"{val:.0f}"
                ax.text(j, i, label, ha="center", va="center",
                       color=color, fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_vo_p95_latency.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'heatmap_vo_p95_latency.png'}")


def main():
    experiments_dir = Path("experiments")
    output_dir = Path("plots/exp2_mix_heatmap")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting experiment parameters...")
    params_df = extract_experiment_params(experiments_dir)
    print(f"Found {len(params_df)} experiments")

    print("\nParameter coverage:")
    print(f"  FA ratios: {sorted(params_df['fa_ratio'].unique())}")
    print(f"  Scales: {sorted(params_df['inter_arrival_scale'].unique())}")

    print("\nLoading experiment results...")
    results_df = load_experiment_results(params_df)
    print(f"Loaded {len(results_df)} experiments with results")

    # Save combined data
    results_df.to_csv(output_dir / "heatmap_data.csv", index=False)
    print(f"\nSaved heatmap data to {output_dir / 'heatmap_data.csv'}")

    # Create heatmaps
    print("\nGenerating heatmaps...")

    # 1. Success Rate
    create_heatmap(
        results_df, "success_rate",
        "Success Rate (%) by Operation Mix and Load",
        output_dir / "heatmap_success_rate.png",
        cmap="RdYlGn", vmin=60, vmax=100
    )

    # 2. Throughput (commits/hour)
    create_heatmap(
        results_df, "throughput",
        "Throughput (commits/hour) by Operation Mix and Load",
        output_dir / "heatmap_throughput.png",
        cmap="plasma", fmt=".0f"
    )

    # 3. Mean Latency
    create_heatmap(
        results_df, "mean_latency",
        "Mean Commit Latency (ms) by Operation Mix and Load",
        output_dir / "heatmap_mean_latency.png",
        cmap="inferno_r", fmt=".0f"
    )

    # 4. P99 Latency
    create_heatmap(
        results_df, "p99_latency",
        "P99 Commit Latency (ms) by Operation Mix and Load",
        output_dir / "heatmap_p99_latency.png",
        cmap="inferno_r", fmt=".0f"
    )

    # 5. Per-operation-type heatmaps (VO-specific)
    print("\nGenerating per-operation-type heatmaps...")
    generate_vo_heatmaps(experiments_dir, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
