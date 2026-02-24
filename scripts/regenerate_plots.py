#!/usr/bin/env python3
"""Unified plot regeneration from experiment configs.

Scans experiment_configs/*.toml for [plots] sections, dispatches to the
appropriate plotting function for each [[plots.graphs]] entry, and merges
per-graph overrides with plotting.toml defaults.

Usage:
    python scripts/regenerate_plots.py [options]

Options:
    --parallel N       Concurrent workers (default: 4)
    --config PATH      Process single experiment config
    --pattern GLOB     Only configs matching pattern (e.g., "exp3*")
    --dry-run          Show what would be generated
    --input-dir DIR    Override experiments base directory
"""

import argparse
import os
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Graph registry: type name -> (function_name, data_pipeline)
# data_pipeline is "index" (needs index_df) or "raw" (needs base_dir/pattern)
# ---------------------------------------------------------------------------

GRAPH_REGISTRY = {
    "latency_vs_throughput":       ("plot_latency_vs_throughput", "index"),
    "latency_vs_throughput_table": ("generate_latency_vs_throughput_table", "index"),
    "success_rate_vs_load":        ("plot_success_rate_vs_load", "index"),
    "success_rate_vs_throughput":  ("plot_success_rate_vs_throughput", "index"),
    "overhead_vs_throughput":      ("plot_overhead_vs_throughput", "index"),
    "commit_rate_over_time":       ("plot_commit_rate_over_time", "time_series"),
    "sustainable_throughput":      ("plot_sustainable_throughput", "index"),
    "heatmap":                     ("generate_heatmap_plots", "raw"),
    "operation_types":             ("generate_operation_type_plots", "raw"),
}

# Output filenames per graph type
OUTPUT_FILES = {
    "latency_vs_throughput":       "latency_vs_throughput.png",
    "latency_vs_throughput_table": "latency_vs_throughput.md",
    "success_rate_vs_load":        "success_vs_load.png",
    "success_rate_vs_throughput":  "success_vs_throughput.png",
    "overhead_vs_throughput":      "overhead_vs_throughput.png",
    "commit_rate_over_time":       "commit_rate_over_time.png",
    "sustainable_throughput":      "sustainable_throughput.png",
    # heatmap and operation_types handle their own output files
}


def load_plotting_defaults() -> dict:
    """Load plotting.toml defaults."""
    plotting_path = Path("plotting.toml")
    if not plotting_path.exists():
        print("Warning: plotting.toml not found, using empty defaults")
        return {}

    with open(plotting_path, "rb") as f:
        return tomllib.load(f)


def find_experiment_configs(pattern: str = None, config_path: str = None) -> list[Path]:
    """Find experiment configs with [plots] sections."""
    if config_path:
        return [Path(config_path)]

    config_dir = Path("experiment_configs")
    if not config_dir.exists():
        print("Error: experiment_configs/ directory not found")
        return []

    glob_pattern = f"{pattern}.toml" if pattern else "*.toml"
    configs = []

    for path in sorted(config_dir.glob(glob_pattern)):
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        if "plots" in cfg:
            configs.append(path)

    return configs


def verify_seed_consistency(experiments_dir: str, pattern: str) -> bool:
    """Check that all experiments matching pattern have consistent seed counts."""
    base_dir = Path(experiments_dir)
    seed_counts = {}

    for exp_dir in sorted(base_dir.glob(pattern)):
        if not exp_dir.is_dir():
            continue
        seeds = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        seed_counts[exp_dir.name] = len(seeds)

    if not seed_counts:
        return True

    counts = list(seed_counts.values())
    majority = max(set(counts), key=counts.count)
    inconsistent = {name: c for name, c in seed_counts.items() if c != majority}

    if inconsistent:
        print(f"  Warning: seed inconsistency (majority={majority} seeds):")
        for name, count in sorted(inconsistent.items()):
            print(f"    {name}: {count} seeds")
        return False

    return True


def process_config(config_path: Path, plotting_defaults: dict,
                   input_dir: str = "experiments", dry_run: bool = False) -> dict:
    """Process a single experiment config and generate its plots.

    Returns dict with status info.
    """
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)

    label = cfg.get("experiment", {}).get("label", config_path.stem)
    plots_cfg = cfg.get("plots", {})
    output_dir = plots_cfg.get("output_dir", f"plots/{label}")
    pattern = plots_cfg.get("pattern", f"{label}-*")
    graphs = plots_cfg.get("graphs", [])

    if not graphs:
        return {"config": str(config_path), "label": label, "status": "skipped",
                "reason": "no graphs defined"}

    result = {"config": str(config_path), "label": label, "graphs": [],
              "status": "ok"}

    if dry_run:
        print(f"\n{label} ({config_path}):")
        print(f"  pattern: {pattern}")
        print(f"  output_dir: {output_dir}")
        for graph in graphs:
            graph_type = graph.get("type", "unknown")
            if graph_type not in GRAPH_REGISTRY:
                print(f"  [{graph_type}] UNKNOWN TYPE")
                continue
            output_file = OUTPUT_FILES.get(graph_type, f"{graph_type}/")
            extra = ""
            if "group_by" in graph:
                extra += f" group_by={graph['group_by']}"
            if "filters" in graph:
                extra += f" filters={graph['filters']}"
            print(f"  [{graph_type}] -> {output_dir}/{output_file}{extra}")
        return result

    # Check experiments exist
    base_dir = Path(input_dir)
    matching = list(base_dir.glob(pattern))
    if not matching:
        return {"config": str(config_path), "label": label, "status": "skipped",
                "reason": f"no experiments matching {pattern}"}

    print(f"\n{'='*60}")
    print(f"  {label} ({len(matching)} experiment dirs)")
    print(f"{'='*60}")

    verify_seed_consistency(input_dir, pattern)

    os.makedirs(output_dir, exist_ok=True)

    # Lazy-load analysis module (heavy imports)
    from endive import saturation_analysis as sa

    # Build index_df once for index-based graphs
    index_df = None
    needs_index = any(
        GRAPH_REGISTRY.get(g.get("type", ""), ("", ""))[1] == "index"
        for g in graphs
    )
    if needs_index:
        print(f"  Building experiment index for {pattern}...")
        index_df = sa.build_experiment_index(input_dir, pattern)
        if len(index_df) == 0:
            print(f"  Warning: empty experiment index for {pattern}")

        # Apply filters if any graph has them
        # Save filtered versions per-graph later if needed

    for graph in graphs:
        graph_type = graph.get("type")
        if graph_type not in GRAPH_REGISTRY:
            print(f"  Unknown graph type: {graph_type}")
            result["graphs"].append({"type": graph_type, "status": "error",
                                     "reason": "unknown type"})
            continue

        func_name, pipeline = GRAPH_REGISTRY[graph_type]
        func = getattr(sa, func_name)

        # Merge defaults: plotting.toml[graph_type] <- per-graph overrides
        defaults = dict(plotting_defaults.get(graph_type, {}))
        overrides = {k: v for k, v in graph.items() if k != "type"}
        merged = {**defaults, **overrides}

        # Handle per-graph filters
        graph_index_df = index_df
        if "filters" in merged and graph_index_df is not None:
            graph_index_df = sa.apply_filters(graph_index_df, merged.pop("filters"))

        # Determine output suffix for filtered views
        output_suffix = merged.pop("output_suffix", None)
        graph_output_dir = os.path.join(output_dir, output_suffix) if output_suffix else output_dir
        if output_suffix:
            os.makedirs(graph_output_dir, exist_ok=True)

        try:
            if pipeline == "index":
                output_file = OUTPUT_FILES.get(graph_type, f"{graph_type}.png")
                output_path = os.path.join(graph_output_dir, output_file)

                # Build kwargs from merged config
                kwargs = {}
                if "title" in merged:
                    kwargs["title"] = merged["title"]
                if "group_by" in merged:
                    kwargs["group_by"] = merged["group_by"]
                if "success_threshold" in merged:
                    kwargs["success_threshold"] = merged["success_threshold"]
                if "annotate_success_rate" in merged:
                    kwargs["annotate_success_rate"] = merged["annotate_success_rate"]

                # Allow per-graph output file override
                if "output_file" in merged:
                    output_path = os.path.join(graph_output_dir, merged["output_file"])

                func(graph_index_df, output_path, **kwargs)

            elif pipeline == "time_series":
                output_file = OUTPUT_FILES.get(graph_type, f"{graph_type}.png")
                output_path = os.path.join(graph_output_dir, output_file)

                kwargs = {}
                if "title" in merged:
                    kwargs["title"] = merged["title"]
                if "window_size_sec" in merged:
                    kwargs["window_size_sec"] = merged["window_size_sec"]

                func(input_dir, pattern, output_path, **kwargs)

            elif pipeline == "raw":
                # heatmap and operation_types manage their own output files
                kwargs = {}
                if graph_type == "heatmap":
                    kwargs["x_param"] = merged.get("x_param")
                    kwargs["y_param"] = merged.get("y_param")
                    kwargs["metrics"] = merged.get("metrics", [])
                    kwargs["config"] = merged
                elif graph_type == "operation_types":
                    kwargs["load_levels"] = merged.get("load_levels")
                    kwargs["group_by"] = merged.get("group_by")
                    kwargs["config"] = merged

                func(input_dir, pattern, graph_output_dir, **kwargs)

            result["graphs"].append({"type": graph_type, "status": "ok"})
            print(f"  [{graph_type}] OK")

        except Exception as e:
            result["graphs"].append({"type": graph_type, "status": "error",
                                     "reason": str(e)})
            print(f"  [{graph_type}] ERROR: {e}")

    return result


def _process_config_wrapper(args):
    """Wrapper for ProcessPoolExecutor (must be picklable)."""
    config_path, plotting_defaults, input_dir = args
    return process_config(config_path, plotting_defaults, input_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from experiment configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parallel", "-p", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Process a single experiment config file")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Only process configs matching pattern (e.g., 'exp3*')")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be generated")
    parser.add_argument("--input-dir", "-i", type=str, default="experiments",
                        help="Experiments base directory (default: experiments)")

    args = parser.parse_args()

    print("=" * 60)
    print("  ENDIVE PLOT REGENERATION")
    print("=" * 60)

    # Load global plotting defaults
    plotting_defaults = load_plotting_defaults()

    # Find configs to process
    configs = find_experiment_configs(pattern=args.pattern, config_path=args.config)
    if not configs:
        print("No experiment configs with [plots] sections found.")
        return

    print(f"Found {len(configs)} config(s) with [plots] sections:")
    for c in configs:
        print(f"  {c}")

    if args.dry_run:
        for config_path in configs:
            process_config(config_path, plotting_defaults, args.input_dir, dry_run=True)
        return

    # Process configs (sequentially for now — each config can be I/O heavy)
    # Parallelism is within-config if needed
    results = []
    for config_path in configs:
        result = process_config(config_path, plotting_defaults, args.input_dir)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        label = r.get("label", "?")
        status = r.get("status", "?")
        if status == "skipped":
            print(f"  {label}: SKIPPED ({r.get('reason', '')})")
        else:
            graphs = r.get("graphs", [])
            ok = sum(1 for g in graphs if g.get("status") == "ok")
            err = sum(1 for g in graphs if g.get("status") == "error")
            print(f"  {label}: {ok} OK, {err} errors")
    print()


if __name__ == "__main__":
    main()
