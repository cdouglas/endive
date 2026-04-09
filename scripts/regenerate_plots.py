#!/usr/bin/env python3
"""Unified plot regeneration from experiment configs.

Scans experiment_configs/*.toml for [plots] sections, dispatches to the
appropriate plotting function for each [[plots.graphs]] entry, and merges
per-graph overrides with plotting.toml defaults.

Usage:
    python scripts/regenerate_plots.py [options]

Options:
    --parallel N          Concurrent workers across configs (default: 4)
    --intra-parallel N    Concurrent workers within a config (default: 1)
    --config PATH         Process single experiment config
    --pattern GLOB        Only configs matching pattern (e.g., "exp3*")
    --dry-run             Show what would be generated
    --input-dir DIR       Override experiments base directory
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
    "per_table_breakdown":         ("generate_per_table_plots", "raw"),
    "workload_knee_table":         ("generate_workload_knee_table", "raw"),
    "compare":                     (None, "compare"),
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

# Companion markdown tables auto-emitted alongside plots.
# Maps graph type -> (table function name,).
# Skips heatmap/operation_types (they already emit CSV data files).
TABLE_COMPANIONS = {
    "latency_vs_throughput":      "generate_latency_vs_throughput_table",
    "success_rate_vs_throughput": "generate_success_rate_vs_throughput_table",
    "commit_rate_over_time":      "generate_commit_rate_over_time_table",
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


def _process_single_graph(graph, graph_type, plotting_defaults, sa, input_dir,
                          pattern, output_dir, index_df, experiments_cache):
    """Process a single graph entry. Returns (graph_type, status_dict).

    Factored out of process_config() to enable parallel dispatch.
    """
    func_name, pipeline = GRAPH_REGISTRY[graph_type]
    func = getattr(sa, func_name) if func_name else None

    # Merge defaults: plotting.toml[graph_type] <- per-graph overrides
    defaults = dict(plotting_defaults.get(graph_type, {}))
    overrides = {k: v for k, v in graph.items() if k != "type"}
    merged = {**defaults, **overrides}

    # Handle per-graph filters for index-based graphs
    graph_index_df = index_df
    if "filters" in merged and graph_index_df is not None and pipeline == "index":
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
            elif graph_type == "per_table_breakdown":
                kwargs["config"] = merged
            elif graph_type == "workload_knee_table":
                kwargs["config"] = merged

            # Pass experiments_cache to avoid redundant scans
            kwargs["experiments_cache"] = experiments_cache
            func(input_dir, pattern, graph_output_dir, **kwargs)

        elif pipeline == "compare":
            # Cross-experiment comparison: load two experiment patterns,
            # label them, concatenate, and dispatch to an index-based plot.
            import pandas as pd

            base_type = merged.get("base_type", "latency_vs_throughput")
            patterns = merged.get("patterns", [])
            labels = merged.get("labels", patterns)
            compare_group_by = merged.get("group_by")

            if len(patterns) < 2:
                raise ValueError("compare requires at least 2 patterns")

            # Build index for each pattern and label
            frames = []
            for pat, label in zip(patterns, labels):
                idx = sa.build_experiment_index(input_dir, pat + "*")
                if len(idx) > 0:
                    idx["experiment"] = label
                    frames.append(idx)

            if not frames:
                raise ValueError(f"No data for compare patterns: {patterns}")

            combined = pd.concat(frames, ignore_index=True)

            # Composite group_by: "experiment" alone or "experiment × param"
            if compare_group_by:
                combined["_compare_group"] = (
                    combined["experiment"] + " / "
                    + combined[compare_group_by].astype(str)
                )
                effective_group_by = "_compare_group"
            else:
                effective_group_by = "experiment"

            # Dispatch to the base plot function
            base_func_name = GRAPH_REGISTRY[base_type][0]
            base_func = getattr(sa, base_func_name)
            output_file = merged.get("output_file",
                                     f"compare_{base_type}.png")
            output_path = os.path.join(graph_output_dir, output_file)

            base_kwargs = {}
            if "title" in merged:
                base_kwargs["title"] = merged["title"]
            base_kwargs["group_by"] = effective_group_by

            base_func(combined, output_path, **base_kwargs)

        # Auto-emit companion .md table if registered
        if graph_type in TABLE_COMPANIONS:
            companion_func_name = TABLE_COMPANIONS[graph_type]
            companion_func = getattr(sa, companion_func_name)
            # Filter kwargs to only those accepted by table functions
            table_kwargs = {k: v for k, v in kwargs.items()
                            if k in ("title", "group_by", "window_size_sec")}
            if pipeline == "index":
                md_path = output_path.replace(".png", ".md")
                companion_func(graph_index_df, md_path, **table_kwargs)
            elif pipeline == "time_series":
                md_path = output_path.replace(".png", ".md")
                companion_func(input_dir, pattern, md_path, **table_kwargs)

        print(f"  [{graph_type}] OK")
        return {"type": graph_type, "status": "ok"}

    except Exception as e:
        print(f"  [{graph_type}] ERROR: {e}")
        return {"type": graph_type, "status": "error", "reason": str(e)}


def process_config(config_path: Path, plotting_defaults: dict,
                   input_dir: str = "experiments", dry_run: bool = False,
                   intra_parallel: int = 1) -> dict:
    """Process a single experiment config and generate its plots.

    Args:
        intra_parallel: Number of parallel workers for graphs within this config.
                        Default 1 (sequential). Values > 1 use ThreadPoolExecutor
                        to parallelize independent graph rendering.

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
            if graph_type in TABLE_COMPANIONS:
                md_file = output_file.replace(".png", ".md")
                print(f"  [{graph_type}] -> {output_dir}/{md_file} (companion table)")
        return result

    # Compare-only configs don't need their own experiments
    has_compare_only = all(
        g.get("type") == "compare" for g in graphs
    )

    # Check experiments exist (on disk or in consolidated file)
    base_dir = Path(input_dir)
    matching = list(base_dir.glob(pattern))
    consolidated_path = base_dir / 'consolidated.parquet'
    consolidated_exists = consolidated_path.exists()

    if not matching and not consolidated_exists and not has_compare_only:
        return {"config": str(config_path), "label": label, "status": "skipped",
                "reason": f"no experiments matching {pattern}"}

    if has_compare_only:
        print(f"\n{'='*60}")
        print(f"  {label} (compare-only, {len(graphs)} graphs)")
        print(f"{'='*60}")
    elif not matching and consolidated_exists:
        from endive.saturation_analysis import scan_all_experiments
        consolidated_matches = scan_all_experiments(
            str(base_dir), pattern)
        if not consolidated_matches:
            return {"config": str(config_path), "label": label, "status": "skipped",
                    "reason": f"no experiments matching {pattern}"}
        print(f"\n{'='*60}")
        print(f"  {label} ({len(consolidated_matches)} experiments from consolidated)")
        print(f"{'='*60}")
    else:
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

    # Pre-scan experiments once for all raw-pipeline graphs (Layer 1)
    experiments_cache = None
    needs_raw = any(
        GRAPH_REGISTRY.get(g.get("type", ""), ("", ""))[1] == "raw"
        for g in graphs
    )
    if needs_raw:
        print(f"  Pre-scanning experiments for {pattern}...")
        experiments_cache = sa.scan_all_experiments(input_dir, pattern)
        print(f"  Found {len(experiments_cache)} experiments")

    # Validate graphs
    valid_graphs = []
    for graph in graphs:
        graph_type = graph.get("type")
        if graph_type not in GRAPH_REGISTRY:
            print(f"  Unknown graph type: {graph_type}")
            result["graphs"].append({"type": graph_type, "status": "error",
                                     "reason": "unknown type"})
        else:
            valid_graphs.append((graph, graph_type))

    # Process graphs (parallel or sequential)
    if intra_parallel > 1 and len(valid_graphs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=intra_parallel) as pool:
            futures = {
                pool.submit(
                    _process_single_graph, graph, graph_type, plotting_defaults,
                    sa, input_dir, pattern, output_dir, index_df, experiments_cache
                ): graph_type
                for graph, graph_type in valid_graphs
            }
            for future in as_completed(futures):
                result["graphs"].append(future.result())
    else:
        for graph, graph_type in valid_graphs:
            graph_result = _process_single_graph(
                graph, graph_type, plotting_defaults, sa, input_dir,
                pattern, output_dir, index_df, experiments_cache)
            result["graphs"].append(graph_result)

    return result


def _process_config_wrapper(args):
    """Wrapper for ProcessPoolExecutor (must be picklable)."""
    config_path, plotting_defaults, input_dir, intra_parallel = args
    try:
        return process_config(config_path, plotting_defaults, input_dir,
                              intra_parallel=intra_parallel)
    except Exception as e:
        label = Path(config_path).stem
        return {"config": str(config_path), "label": label,
                "status": "error", "reason": str(e), "graphs": []}


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots from experiment configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--parallel", "-p", type=int, default=4,
                        help="Number of parallel workers across configs (default: 4)")
    parser.add_argument("--intra-parallel", "-j", type=int, default=1,
                        help="Number of parallel workers within a config (default: 1). "
                             "Parallelizes independent graph generation (heatmaps, etc).")
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

    results = []
    if args.parallel > 1 and len(configs) > 1:
        tasks = [(c, plotting_defaults, args.input_dir, args.intra_parallel)
                 for c in configs]
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(_process_config_wrapper, t): t[0] for t in tasks}
            for future in as_completed(futures):
                results.append(future.result())
        # Sort by label for deterministic summary output
        results.sort(key=lambda r: r.get("label", ""))
    else:
        for config_path in configs:
            results.append(process_config(config_path, plotting_defaults,
                                          args.input_dir,
                                          intra_parallel=args.intra_parallel))

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        label = r.get("label", "?")
        status = r.get("status", "?")
        if status == "skipped":
            print(f"  {label}: SKIPPED ({r.get('reason', '')})")
        elif status == "error":
            print(f"  {label}: ERROR ({r.get('reason', '')})")
        else:
            graphs = r.get("graphs", [])
            ok = sum(1 for g in graphs if g.get("status") == "ok")
            err = sum(1 for g in graphs if g.get("status") == "error")
            print(f"  {label}: {ok} OK, {err} errors")
    print()


if __name__ == "__main__":
    main()
