#!/usr/bin/env python
"""
Experiment runner for icecap simulator.

This script generates configuration files for parameter sweeps and runs experiments.
"""

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List


def load_base_config(config_path: str) -> Dict[str, Any]:
    """Load base configuration file."""
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def write_config(config: Dict[str, Any], output_path: str):
    """Write configuration to TOML file."""
    with open(output_path, "w") as f:
        # Write simulation section
        f.write("[simulation]\n")
        f.write(f"duration_ms = {config['simulation']['duration_ms']}\n")
        f.write(f'output_path = "{config["simulation"]["output_path"]}"\n')
        seed = config['simulation'].get('seed')
        if seed is not None:
            f.write(f"seed = {seed}\n")
        else:
            f.write("# seed = 42  # Uncomment to use fixed seed\n")
        f.write("\n")

        # Write catalog section
        f.write("[catalog]\n")
        f.write(f"num_tables = {config['catalog']['num_tables']}\n")
        f.write(f"num_groups = {config['catalog'].get('num_groups', 1)}\n")
        f.write(f'group_size_distribution = "{config["catalog"].get("group_size_distribution", "uniform")}"\n')
        f.write("\n")

        # Write longtail parameters if present
        if "longtail" in config["catalog"]:
            f.write(f"longtail.large_group_fraction = {config['catalog']['longtail'].get('large_group_fraction', 0.5)}\n")
            f.write(f"longtail.medium_groups_count = {config['catalog']['longtail'].get('medium_groups_count', 3)}\n")
            f.write(f"longtail.medium_group_fraction = {config['catalog']['longtail'].get('medium_group_fraction', 0.3)}\n")
            f.write("\n")

        # Write transaction section
        f.write("[transaction]\n")
        f.write(f"retry = {config['transaction']['retry']}\n")
        f.write("runtime.min = {}\n".format(config['transaction']['runtime']['min']))
        f.write("runtime.mean = {}\n".format(config['transaction']['runtime']['mean']))
        f.write("runtime.sigma = {}\n".format(config['transaction']['runtime']['sigma']))
        f.write("\n")

        # Write inter-arrival distribution
        f.write('inter_arrival.distribution = "{}"\n'.format(
            config['transaction']['inter_arrival']['distribution']
        ))
        for key, value in config['transaction']['inter_arrival'].items():
            if key != 'distribution':
                f.write(f"inter_arrival.{key} = {value}\n")
        f.write("\n")

        # Write table distribution parameters
        f.write("ntable.zipf = {}\n".format(config['transaction']['ntable']['zipf']))
        f.write("seltbl.zipf = {}\n".format(config['transaction']['seltbl']['zipf']))
        f.write("seltblw.zipf = {}\n".format(config['transaction']['seltblw']['zipf']))
        f.write("\n")

        # Write storage section
        f.write("[storage]\n")
        f.write(f"max_parallel = {config['storage']['max_parallel']}\n")
        f.write(f"min_latency = {config['storage']['min_latency']}\n")
        f.write("\n")

        # CAS latency
        f.write(f"T_CAS.mean = {config['storage']['T_CAS']['mean']}\n")
        f.write(f"T_CAS.stddev = {config['storage']['T_CAS']['stddev']}\n")
        f.write("\n")

        # Metadata root latencies
        f.write(f"T_METADATA_ROOT.read.mean = {config['storage']['T_METADATA_ROOT']['read']['mean']}\n")
        f.write(f"T_METADATA_ROOT.read.stddev = {config['storage']['T_METADATA_ROOT']['read']['stddev']}\n")
        f.write(f"T_METADATA_ROOT.write.mean = {config['storage']['T_METADATA_ROOT']['write']['mean']}\n")
        f.write(f"T_METADATA_ROOT.write.stddev = {config['storage']['T_METADATA_ROOT']['write']['stddev']}\n")
        f.write("\n")

        # Manifest list latencies
        f.write(f"T_MANIFEST_LIST.read.mean = {config['storage']['T_MANIFEST_LIST']['read']['mean']}\n")
        f.write(f"T_MANIFEST_LIST.read.stddev = {config['storage']['T_MANIFEST_LIST']['read']['stddev']}\n")
        f.write(f"T_MANIFEST_LIST.write.mean = {config['storage']['T_MANIFEST_LIST']['write']['mean']}\n")
        f.write(f"T_MANIFEST_LIST.write.stddev = {config['storage']['T_MANIFEST_LIST']['write']['stddev']}\n")
        f.write("\n")

        # Manifest file latencies
        f.write(f"T_MANIFEST_FILE.read.mean = {config['storage']['T_MANIFEST_FILE']['read']['mean']}\n")
        f.write(f"T_MANIFEST_FILE.read.stddev = {config['storage']['T_MANIFEST_FILE']['read']['stddev']}\n")
        f.write(f"T_MANIFEST_FILE.write.mean = {config['storage']['T_MANIFEST_FILE']['write']['mean']}\n")
        f.write(f"T_MANIFEST_FILE.write.stddev = {config['storage']['T_MANIFEST_FILE']['write']['stddev']}\n")


def sweep_inter_arrival(
    base_config: Dict[str, Any],
    output_dir: str,
    inter_arrival_times: List[float],
    distribution: str = "exponential"
):
    """
    Generate configs sweeping over inter-arrival times (client load).

    Lower inter-arrival time = higher load (more concurrent clients)
    """
    configs = []
    for ia_time in inter_arrival_times:
        config = dict(base_config)
        config['transaction']['inter_arrival']['distribution'] = distribution
        if distribution == "exponential":
            config['transaction']['inter_arrival']['scale'] = ia_time
        elif distribution == "fixed":
            config['transaction']['inter_arrival']['value'] = ia_time

        # Update output path
        config['simulation']['output_path'] = f"{output_dir}/ia_{int(ia_time)}.parquet"

        # Write config file
        config_path = f"{output_dir}/cfg_ia_{int(ia_time)}.toml"
        write_config(config, config_path)
        configs.append((config_path, config['simulation']['output_path']))

    return configs


def sweep_catalog_latency(
    base_config: Dict[str, Any],
    output_dir: str,
    cas_latencies: List[int]
):
    """Generate configs sweeping over catalog CAS latency."""
    configs = []
    for cas_latency in cas_latencies:
        config = dict(base_config)
        # Update mean CAS latency (keep stddev proportional)
        config['storage']['T_CAS']['mean'] = cas_latency
        config['storage']['T_CAS']['stddev'] = cas_latency * 0.1  # 10% stddev

        # Update output path
        config['simulation']['output_path'] = f"{output_dir}/cas_{cas_latency}.parquet"

        # Write config file
        config_path = f"{output_dir}/cfg_cas_{cas_latency}.toml"
        write_config(config, config_path)
        configs.append((config_path, config['simulation']['output_path']))

    return configs


def sweep_combined(
    base_config: Dict[str, Any],
    output_dir: str,
    inter_arrival_times: List[float],
    cas_latencies: List[int],
    distribution: str = "exponential"
):
    """Generate configs sweeping over both inter-arrival times and CAS latencies."""
    configs = []
    for ia_time in inter_arrival_times:
        for cas_latency in cas_latencies:
            config = dict(base_config)
            config['transaction']['inter_arrival']['distribution'] = distribution
            if distribution == "exponential":
                config['transaction']['inter_arrival']['scale'] = ia_time
            elif distribution == "fixed":
                config['transaction']['inter_arrival']['value'] = ia_time

            # Update mean CAS latency (keep stddev proportional)
            config['storage']['T_CAS']['mean'] = cas_latency
            config['storage']['T_CAS']['stddev'] = cas_latency * 0.1  # 10% stddev

            # Update output path
            config['simulation']['output_path'] = (
                f"{output_dir}/ia_{int(ia_time)}_cas_{cas_latency}.parquet"
            )

            # Write config file
            config_path = f"{output_dir}/cfg_ia_{int(ia_time)}_cas_{cas_latency}.toml"
            write_config(config, config_path)
            configs.append((config_path, config['simulation']['output_path']))

    return configs


def run_experiment(config_path: str, verbose: bool = False):
    """Run a single experiment with the given config."""
    cmd = ["python", "-m", "icecap.main", config_path, "--yes", "--no-progress"]
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    print(f"Running experiment: {config_path}")
    result = subprocess.run(cmd, capture_output=not verbose)

    if result.returncode != 0:
        print(f"ERROR: Experiment failed: {config_path}")
        if not verbose:
            print(result.stderr.decode())
        return False
    return True


def cli():
    parser = argparse.ArgumentParser(
        description="Generate and run icecap experiments"
    )
    parser.add_argument(
        "-b", "--base-config",
        default="cfg.toml",
        help="Base configuration file (default: cfg.toml)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="experiments",
        help="Output directory for configs and results (default: experiments)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output during simulation runs"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs but don't run experiments"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inter-arrival sweep
    ia_parser = subparsers.add_parser(
        "sweep-clients",
        help="Sweep over inter-arrival times (client load)"
    )
    ia_parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000, 10000],
        help="Inter-arrival times in ms (default: 100 500 1000 2000 5000 10000)"
    )
    ia_parser.add_argument(
        "--dist",
        choices=["exponential", "fixed"],
        default="exponential",
        help="Inter-arrival distribution (default: exponential)"
    )

    # CAS latency sweep
    cas_parser = subparsers.add_parser(
        "sweep-latency",
        help="Sweep over catalog CAS latency"
    )
    cas_parser.add_argument(
        "--latencies",
        type=int,
        nargs="+",
        default=[10, 50, 100, 200, 500, 1000],
        help="CAS latencies in ms (default: 10 50 100 200 500 1000)"
    )

    # Combined sweep
    combined_parser = subparsers.add_parser(
        "sweep-combined",
        help="Sweep over both client load and catalog latency"
    )
    combined_parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[500, 1000, 5000],
        help="Inter-arrival times in ms (default: 500 1000 5000)"
    )
    combined_parser.add_argument(
        "--latencies",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="CAS latencies in ms (default: 50 100 200)"
    )
    combined_parser.add_argument(
        "--dist",
        choices=["exponential", "fixed"],
        default="exponential",
        help="Inter-arrival distribution (default: exponential)"
    )

    args = parser.parse_args()

    # Load base config
    base_config = load_base_config(args.base_config)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate configs based on command
    if args.command == "sweep-clients":
        configs = sweep_inter_arrival(
            base_config, args.output_dir, args.times, args.dist
        )
    elif args.command == "sweep-latency":
        configs = sweep_catalog_latency(
            base_config, args.output_dir, args.latencies
        )
    elif args.command == "sweep-combined":
        configs = sweep_combined(
            base_config, args.output_dir, args.times, args.latencies, args.dist
        )

    print(f"Generated {len(configs)} experiment configurations")

    # Run experiments
    if not args.dry_run:
        success = 0
        for config_path, _ in configs:
            if run_experiment(config_path, args.verbose):
                success += 1
        print(f"\nCompleted {success}/{len(configs)} experiments successfully")
    else:
        print("Dry run - configs generated but not executed")


if __name__ == "__main__":
    cli()
