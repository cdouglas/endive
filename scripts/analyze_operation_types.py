#!/usr/bin/env python3
"""Analyze per-operation-type metrics from exp2_mix_heatmap experiments."""

import os
import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_experiments(experiments_dir: Path) -> pd.DataFrame:
    """Extract per-operation-type metrics from all experiments."""
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

        # Process each seed
        for seed_dir in exp_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            results_path = seed_dir / "results.parquet"
            if not results_path.exists():
                continue

            df = pd.read_parquet(results_path)

            # Per operation type stats
            for op_type in ["fast_append", "validated_overwrite"]:
                op_df = df[df["operation_type"] == op_type]
                if len(op_df) == 0:
                    continue

                committed = op_df[op_df["status"].str.lower() == "committed"]
                aborted = op_df[op_df["status"].str.lower() == "aborted"]

                records.append({
                    "hash": exp_dir.name.split("-")[-1],
                    "seed": seed_dir.name,
                    "fa_ratio": fa_ratio,
                    "inter_arrival_scale": scale,
                    "operation_type": op_type,
                    "total": len(op_df),
                    "committed": len(committed),
                    "aborted": len(aborted),
                    "success_rate": len(committed) / len(op_df) * 100 if len(op_df) > 0 else 0,
                    "mean_latency": committed["commit_latency"].mean() if len(committed) > 0 else np.nan,
                    "p95_latency": committed["commit_latency"].quantile(0.95) if len(committed) > 0 else np.nan,
                    "mean_retries": committed["n_retries"].mean() if len(committed) > 0 else np.nan,
                })

    return pd.DataFrame(records)


def main():
    experiments_dir = Path("experiments")
    output_dir = Path("plots/exp2_mix_heatmap")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing per-operation-type metrics...")
    df = analyze_experiments(experiments_dir)
    print(f"Collected {len(df)} records")

    # Aggregate by experiment (across seeds)
    agg = df.groupby(["hash", "fa_ratio", "inter_arrival_scale", "operation_type"]).agg({
        "total": "sum",
        "committed": "sum",
        "aborted": "sum",
        "success_rate": "mean",
        "mean_latency": "mean",
        "p95_latency": "mean",
        "mean_retries": "mean",
    }).reset_index()

    agg["success_rate"] = agg["committed"] / agg["total"] * 100

    # Save detailed data
    agg.to_csv(output_dir / "operation_type_metrics.csv", index=False)
    print(f"Saved to {output_dir / 'operation_type_metrics.csv'}")

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter to specific load levels for clarity
    for idx, scale in enumerate([20, 100, 500, 2000]):
        ax = axes[idx // 2, idx % 2]
        subset = agg[agg["inter_arrival_scale"] == scale]

        for op_type, color, marker in [("fast_append", "blue", "o"), ("validated_overwrite", "red", "s")]:
            op_data = subset[subset["operation_type"] == op_type].sort_values("fa_ratio")
            if len(op_data) > 0:
                ax.plot(op_data["fa_ratio"], op_data["success_rate"],
                       color=color, marker=marker, label=op_type.replace("_", " ").title(),
                       linewidth=2, markersize=6)

        ax.set_xlabel("FastAppend Ratio")
        ax.set_ylabel("Success Rate (%)")
        ax.set_title(f"Scale = {scale}ms (TPS ≈ {1000/scale:.1f})")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(50, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Success Rate by Operation Type Across Different Loads", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "op_type_success_by_load.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'op_type_success_by_load.png'}")

    # Latency comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, scale in enumerate([20, 100, 500, 2000]):
        ax = axes[idx // 2, idx % 2]
        subset = agg[agg["inter_arrival_scale"] == scale]

        for op_type, color, marker in [("fast_append", "blue", "o"), ("validated_overwrite", "red", "s")]:
            op_data = subset[subset["operation_type"] == op_type].sort_values("fa_ratio")
            if len(op_data) > 0:
                ax.plot(op_data["fa_ratio"], op_data["mean_latency"],
                       color=color, marker=marker, label=op_type.replace("_", " ").title(),
                       linewidth=2, markersize=6)

        ax.set_xlabel("FastAppend Ratio")
        ax.set_ylabel("Mean Commit Latency (ms)")
        ax.set_title(f"Scale = {scale}ms (TPS ≈ {1000/scale:.1f})")
        ax.set_xlim(-0.05, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Commit Latency by Operation Type Across Different Loads", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "op_type_latency_by_load.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'op_type_latency_by_load.png'}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Per-Operation-Type Behavior")
    print("="*60)

    fa_data = agg[agg["operation_type"] == "fast_append"]
    vo_data = agg[agg["operation_type"] == "validated_overwrite"]

    print(f"\nFastAppend:")
    print(f"  Success rate: {fa_data['success_rate'].min():.1f}% - {fa_data['success_rate'].max():.1f}%")
    print(f"  Mean latency: {fa_data['mean_latency'].min():.0f}ms - {fa_data['mean_latency'].max():.0f}ms")
    print(f"  Total aborts: {fa_data['aborted'].sum():.0f} / {fa_data['total'].sum():.0f}")

    print(f"\nValidatedOverwrite:")
    print(f"  Success rate: {vo_data['success_rate'].min():.1f}% - {vo_data['success_rate'].max():.1f}%")
    print(f"  Mean latency: {vo_data['mean_latency'].min():.0f}ms - {vo_data['mean_latency'].max():.0f}ms")
    print(f"  Total aborts: {vo_data['aborted'].sum():.0f} / {vo_data['total'].sum():.0f}")

    # Key finding: VO at high FA ratio
    high_fa_vo = vo_data[vo_data["fa_ratio"] >= 0.9]
    if len(high_fa_vo) > 0:
        print(f"\nVO at high FA ratio (>=0.9):")
        print(f"  Success rate: {high_fa_vo['success_rate'].mean():.1f}%")
        print(f"  Mean latency: {high_fa_vo['mean_latency'].mean():.0f}ms")


if __name__ == "__main__":
    main()
