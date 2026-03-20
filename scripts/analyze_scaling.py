#!/usr/bin/env python3
"""Analyze multi-table scaling and cross-experiment comparisons.

Reads heatmap_data.csv and per_table_data.csv files from experiment plot
directories and produces summary tables for reports.

Usage:
    # Provider scaling comparison (exp4c)
    python scripts/analyze_scaling.py provider-scaling

    # Zipf vs Uniform comparison (exp4a vs exp4a-zipf)
    python scripts/analyze_scaling.py zipf-comparison

    # Per-table breakdown under Zipf
    python scripts/analyze_scaling.py zipf-per-table

    # All analyses
    python scripts/analyze_scaling.py all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def _load_heatmap_csv(experiment: str, subdir: str) -> pd.DataFrame | None:
    """Load heatmap_data.csv from plots/<experiment>/<subdir>/."""
    path = PLOTS_DIR / experiment / subdir / "heatmap_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_per_table_csv(experiment: str, subdir: str) -> pd.DataFrame | None:
    """Load per_table_data.csv from plots/<experiment>/<subdir>/."""
    path = PLOTS_DIR / experiment / subdir / "per_table_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _find_subdirs(experiment: str) -> list[str]:
    """Find all subdirectories with heatmap_data.csv."""
    exp_dir = PLOTS_DIR / experiment
    if not exp_dir.exists():
        return []
    return sorted(
        d.name for d in exp_dir.iterdir()
        if d.is_dir() and (d / "heatmap_data.csv").exists()
    )


# ---------------------------------------------------------------------------
# Provider scaling analysis (exp4c)
# ---------------------------------------------------------------------------

def provider_scaling():
    """Show how each provider scales with table count."""
    print("# Provider Scaling Analysis (exp4c)\n")

    # Load workload knee data if available
    knee_path = PLOTS_DIR / "exp4c_tables_providers" / "workload_knee" / "workload_knee_data.csv"
    if knee_path.exists():
        df = pd.read_csv(knee_path)
        _provider_scaling_from_knee(df)
        return

    # Fall back to heatmap data
    subdirs = _find_subdirs("exp4c_tables_providers")
    if not subdirs:
        print("No exp4c heatmap data found.")
        return

    # Load all provider FA=100% heatmaps
    providers = ["s3x", "s3", "azurex", "azure", "gcp"]
    for provider in providers:
        subdir = f"{provider}_fa100"
        df = _load_heatmap_csv("exp4c_tables_providers", subdir)
        if df is None:
            continue
        _print_provider_scaling(provider, df)


def _compute_knee(prov_df: pd.DataFrame, op_type: str = "fast_append") -> dict[int, float]:
    """Compute workload knee (max c/s at >95% success) per num_tables.

    workload_knee_data.csv stores success_rate as 0-100 (percentage).
    """
    result = {}
    op_df = prov_df[prov_df["operation_type"] == op_type]
    for nt in sorted(op_df["num_tables"].unique()):
        nt_df = op_df[op_df["num_tables"] == nt]
        knees = []
        for ias, grp in nt_df.groupby("inter_arrival_scale"):
            sr = grp["success_rate"].iloc[0]  # 0-100 percentage
            tput = grp["committed"].sum() / grp["active_window_s"].iloc[0]
            if sr >= 95.0:
                knees.append(tput)
        result[nt] = max(knees) if knees else None
    return result


def _provider_scaling_from_knee(df: pd.DataFrame):
    """Compute scaling from workload knee CSV data."""
    providers = ["s3x", "s3", "azurex", "azure", "gcp"]
    table_counts = [1, 2, 5, 10, 20, 50]

    # FA=100% knee table
    fa100 = df[df["fa_ratio"] == 1.0] if "fa_ratio" in df.columns else df

    print("## FA=100% Throughput Knee (c/s)\n")
    header = "| Provider |" + "|".join(f" {t}t " for t in table_counts) + "| Scale 1->50 |"
    sep = "|----------|" + "|".join("------:" for _ in table_counts) + "|------------|"
    print(header)
    print(sep)

    for provider in providers:
        prov_df = fa100[fa100["storage_provider"] == provider]
        if prov_df.empty:
            continue
        knees = _compute_knee(prov_df, "fast_append")
        vals = [knees.get(nt) for nt in table_counts]
        if vals[0] and vals[-1]:
            scale = f"{vals[-1] / vals[0]:.1f}x"
        else:
            scale = "—"
        cells = [f" {v:.1f}" if v else " —" for v in vals]
        print(f"| {provider:<8} |{'|'.join(cells)}| {scale:>10} |")

    print()

    # VO latency at knee for mixed workloads
    for ratio_label, ratio_val in [("90/10", 0.9), ("50/50", 0.5)]:
        mixed = df[df["fa_ratio"] == ratio_val] if "fa_ratio" in df.columns else pd.DataFrame()
        if mixed.empty:
            continue
        print(f"## {ratio_label} FA/VO Mix — VO Latency at Knee (seconds)\n")
        header = "| Provider |" + "|".join(f" {t}t " for t in table_counts) + "|"
        sep = "|----------|" + "|".join("------:" for _ in table_counts) + "|"
        print(header)
        print(sep)
        for provider in providers:
            prov_df = mixed[mixed["storage_provider"] == provider]
            if prov_df.empty:
                continue
            # Find knee for FA, then report VO latency at that inter_arrival_scale
            fa_knees = _compute_knee(prov_df, "fast_append")
            vo_df = prov_df[prov_df["operation_type"] == "validated_overwrite"]
            cells = []
            for nt in table_counts:
                nt_vo = vo_df[vo_df["num_tables"] == nt]
                # Find the ias with highest committed VO at >95% FA success
                nt_fa = prov_df[(prov_df["num_tables"] == nt) & (prov_df["operation_type"] == "fast_append")]
                best_lat = None
                best_tput = 0
                for ias in nt_fa["inter_arrival_scale"].unique():
                    fa_row = nt_fa[nt_fa["inter_arrival_scale"] == ias]
                    vo_row = nt_vo[nt_vo["inter_arrival_scale"] == ias]
                    if fa_row.empty or vo_row.empty:
                        continue
                    fa_sr = fa_row.iloc[0]["success_rate"]  # 0-100
                    vo_committed = vo_row.iloc[0]["committed"]
                    tput = vo_committed / vo_row.iloc[0]["active_window_s"]
                    if fa_sr >= 95.0 and tput > best_tput:
                        best_tput = tput
                        best_lat = vo_row.iloc[0]["mean_latency"] / 1000  # ms -> s
                if best_lat is not None:
                    cells.append(f" {best_lat:.1f}s")
                else:
                    cells.append(" —")
            print(f"| {provider:<8} |{'|'.join(cells)}|")
        print()


def _print_provider_scaling(provider: str, df: pd.DataFrame):
    """Print scaling table for one provider from heatmap data."""
    print(f"## {provider}\n")
    if "num_tables" not in df.columns:
        print("No num_tables column found.\n")
        return

    table_counts = sorted(df["num_tables"].unique())
    print("| Tables | Best Success (ias) | Best Throughput (c/s) | Mean Latency |")
    print("|--------|-------------------|----------------------|-------------|")

    for nt in table_counts:
        sub = df[df["num_tables"] == nt]
        # Find the row with highest throughput at >95% success
        good = sub[sub["success_rate"] >= 95.0]  # heatmap CSVs use 0-100
        if not good.empty:
            best = good.loc[good["throughput"].idxmax()]
            tput = best["throughput"] / 3600  # per hour to per second
            ias = best["inter_arrival_scale"]
            lat = best.get("mean_latency", 0)
            lat_str = f"{lat:.0f}ms" if lat < 1000 else f"{lat / 1000:.2f}s"
            print(f"| {nt:>6} | {ias:>17.0f}ms | {tput:>20.1f} | {lat_str:>11} |")
        else:
            print(f"| {nt:>6} | — | — | — |")
    print()


# ---------------------------------------------------------------------------
# Zipf vs Uniform comparison
# ---------------------------------------------------------------------------

def zipf_comparison():
    """Compare Zipf and Uniform success rates and throughput."""
    print("# Zipf vs Uniform Comparison\n")

    cas_levels = ["cas_1ms", "cas_10ms", "cas_50ms", "cas_120ms"]
    pairs = [
        ("exp4a_tables_fa", "exp4a_zipf_tables_fa", "FA=100%"),
        ("exp4b_tables_mix", "exp4b_zipf_tables_mix", "FA=90% / VO=10%"),
    ]

    for uniform_exp, zipf_exp, label in pairs:
        print(f"## {label}\n")
        for cas in cas_levels:
            df_u = _load_heatmap_csv(uniform_exp, cas)
            df_z = _load_heatmap_csv(zipf_exp, cas)
            if df_u is None or df_z is None:
                continue

            _print_zipf_vs_uniform(cas, df_u, df_z)


def _print_zipf_vs_uniform(cas_label: str, df_u: pd.DataFrame, df_z: pd.DataFrame):
    """Print side-by-side Zipf vs Uniform success rates."""
    print(f"### {cas_label}\n")

    table_counts = sorted(set(df_u["num_tables"].unique()) & set(df_z["num_tables"].unique()))
    ias_values = sorted(set(df_u["inter_arrival_scale"].unique()) & set(df_z["inter_arrival_scale"].unique()))

    # Success rate comparison
    header = "| ias (ms) |" + "|".join(f" {t}t Z/U " for t in table_counts) + "|"
    sep = "|----------|" + "|".join("---------:" for _ in table_counts) + "|"
    print("**Success Rate (%)**\n")
    print(header)
    print(sep)

    for ias in ias_values[:8]:  # limit rows
        cells = []
        for nt in table_counts:
            row_u = df_u[(df_u["num_tables"] == nt) & (df_u["inter_arrival_scale"] == ias)]
            row_z = df_z[(df_z["num_tables"] == nt) & (df_z["inter_arrival_scale"] == ias)]
            if not row_u.empty and not row_z.empty:
                sr_u = row_u.iloc[0]["success_rate"]  # already 0-100
                sr_z = row_z.iloc[0]["success_rate"]
                cells.append(f" {sr_z:.0f}/{sr_u:.0f}")
            else:
                cells.append(" —")
        print(f"| {ias:>8.0f} |{'|'.join(cells)}|")
    print()

    # Throughput comparison
    print("**Throughput (c/s)**\n")
    print(header.replace("Z/U", "Z/U"))
    print(sep)

    for ias in ias_values[:8]:
        cells = []
        for nt in table_counts:
            row_u = df_u[(df_u["num_tables"] == nt) & (df_u["inter_arrival_scale"] == ias)]
            row_z = df_z[(df_z["num_tables"] == nt) & (df_z["inter_arrival_scale"] == ias)]
            if not row_u.empty and not row_z.empty:
                tp_u = row_u.iloc[0]["throughput"] / 3600
                tp_z = row_z.iloc[0]["throughput"] / 3600
                cells.append(f" {tp_z:.1f}/{tp_u:.1f}")
            else:
                cells.append(" —")
        print(f"| {ias:>8.0f} |{'|'.join(cells)}|")
    print()


# ---------------------------------------------------------------------------
# Zipf per-table analysis
# ---------------------------------------------------------------------------

def zipf_per_table():
    """Show per-table breakdown under Zipf distribution."""
    print("# Zipf Per-Table Breakdown\n")

    cas_levels = ["cas_1ms", "cas_120ms"]
    experiments = [
        ("exp4a_zipf_tables_fa", "FA=100%"),
        ("exp4b_zipf_tables_mix", "FA=90% / VO=10%"),
    ]

    for exp, label in experiments:
        print(f"## {label}\n")
        for cas in cas_levels:
            df = _load_per_table_csv(exp, cas)
            if df is None:
                continue
            _print_per_table_breakdown(cas, df)


def _print_per_table_breakdown(cas_label: str, df: pd.DataFrame):
    """Print per-table success/latency for a Zipf experiment."""
    print(f"### {cas_label}\n")

    if "num_tables" not in df.columns:
        print("Missing num_tables column.\n")
        return

    # Pick a representative table count and load level
    table_counts = sorted(df["num_tables"].unique())
    ias_values = sorted(df["inter_arrival_scale"].unique())

    # Find a moderate load (first ias where overall SR is 50-90%)
    for nt in [10, 20, 50]:
        if nt not in table_counts:
            continue

        sub = df[df["num_tables"] == nt]
        # Pick the operation type column if present
        op_col = "operation_type" if "operation_type" in sub.columns else None

        for ias in ias_values:
            ias_sub = sub[sub["inter_arrival_scale"] == ias]
            if op_col:
                ias_sub = ias_sub[ias_sub[op_col] == "fast_append"]

            if ias_sub.empty:
                continue

            mean_sr = ias_sub["success_rate"].mean()  # 0-100
            if 30 < mean_sr < 95:
                print(f"N={nt}, ias={ias:.0f}ms (mean SR={mean_sr:.1f}%)\n")
                _print_table_ranks(ias_sub)
                print()
                break
        break  # only one table count


def _print_table_ranks(df: pd.DataFrame):
    """Print ranked table statistics."""
    rank_col = "table_rank" if "table_rank" in df.columns else "table_id"
    if rank_col not in df.columns:
        print("No table rank column found.")
        return

    df_sorted = df.sort_values(rank_col)

    cols = ["write_share", "success_rate", "mean_retries", "mean_latency"]
    available = [c for c in cols if c in df.columns]

    header = f"| Rank | " + " | ".join(available) + " |"
    sep = "|------|" + "|".join("--------:" for _ in available) + "|"
    print(header)
    print(sep)

    for _, row in df_sorted.iterrows():
        rank = int(row[rank_col])
        cells = []
        for col in available:
            val = row[col]
            if col == "write_share":
                cells.append(f" {val:.1f}%")  # already 0-100
            elif col == "success_rate":
                cells.append(f" {val:.1f}%")  # already 0-100
            elif col == "mean_latency":
                cells.append(f" {val:.0f}ms" if val < 1000 else f" {val / 1000:.1f}s")
            else:
                cells.append(f" {val:.1f}")
        print(f"| {rank:>4} |{'|'.join(cells)}|")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze multi-table scaling and comparisons")
    parser.add_argument(
        "analysis",
        choices=["provider-scaling", "zipf-comparison", "zipf-per-table", "all"],
        help="Which analysis to run",
    )
    args = parser.parse_args()

    analyses = {
        "provider-scaling": provider_scaling,
        "zipf-comparison": zipf_comparison,
        "zipf-per-table": zipf_per_table,
    }

    if args.analysis == "all":
        for name, func in analyses.items():
            func()
            print("\n" + "=" * 72 + "\n")
    else:
        analyses[args.analysis]()


if __name__ == "__main__":
    main()
