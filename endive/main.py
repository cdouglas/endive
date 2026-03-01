#!/usr/bin/env python
"""Endive simulator CLI entry point.

Usage:
    python -m endive.main config.toml --yes

This is the CLI wrapper around the simulator. The actual simulation
is run by endive.simulation.Simulation. Configuration is loaded
by endive.config.load_simulation_config().

Experiment management (directory structure, hashing, skip-if-exists)
is handled here.
"""

import argparse
import logging
import shutil
import sys
import tomllib
from pathlib import Path

import numpy as np
from endive.config import (
    ConfigurationError,
    compute_experiment_hash,
    load_simulation_config,
    validate_config,
)
from endive.simulation import Simulation
from endive.utils import get_git_sha

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment management
# ---------------------------------------------------------------------------

def check_existing_experiment(config: dict, config_file: str) -> tuple[bool, int | None, str | None]:
    """Check if experiment results already exist and are complete.

    Returns:
        (skip, seed, output_path):
            - skip: True if results exist and simulation should be skipped
            - seed: Existing seed if found, None otherwise
            - output_path: Path to existing results if found, None otherwise
    """
    label = config.get("experiment", {}).get("label")

    if label is None:
        # No experiment label - check if simple output file exists
        output_path = config["simulation"]["output_path"]
        if Path(output_path).exists():
            return (True, None, output_path)
        return (False, None, None)

    # Compute experiment hash
    exp_hash = compute_experiment_hash(config)
    exp_dir = Path("experiments") / f"{label}-{exp_hash}"

    # Check if experiment directory exists
    if not exp_dir.exists():
        return (False, None, None)

    # Check for existing cfg.toml and validate hash
    exp_config_path = exp_dir / "cfg.toml"
    if exp_config_path.exists():
        try:
            with open(exp_config_path, "rb") as f:
                existing_config = tomllib.load(f)
            existing_hash = compute_experiment_hash(existing_config)

            if existing_hash != exp_hash:
                logger.warning(f"Hash mismatch in {exp_dir}")
                logger.warning(f"   Expected: {exp_hash}")
                logger.warning(f"   Found:    {existing_hash}")
                logger.warning(f"   Configuration may have changed - continuing anyway")
        except Exception as e:
            logger.warning(f"Could not validate existing config: {e}")

    # Look for completed runs (seed directories with results.parquet)
    output_filename = config["simulation"]["output_path"]
    completed_seeds = []

    if exp_dir.exists():
        for seed_dir in exp_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.isdigit():
                results_path = seed_dir / output_filename
                if results_path.exists():
                    completed_seeds.append(int(seed_dir.name))

    # Check if configured seed is already completed
    configured_seed = config.get("simulation", {}).get("seed")
    if configured_seed is not None and configured_seed in completed_seeds:
        output_path = exp_dir / str(configured_seed) / output_filename
        return (True, configured_seed, str(output_path))

    return (False, None, None)


def prepare_experiment_output(config: dict, config_file: str, actual_seed: int) -> str:
    """Prepare experiment output directory and return final output path.

    If experiment.label is set:
    - Creates: experiments/$label-$hash/$seed/
    - Writes: experiments/$label-$hash/cfg.toml (copy of input config)
    - Returns: experiments/$label-$hash/$seed/results.parquet

    If experiment.label is not set:
    - Returns: original output_path from config

    Args:
        config: Parsed TOML configuration dict
        config_file: Path to original config file
        actual_seed: The actual seed being used (either from config or randomly generated)

    Returns:
        Final output path for results
    """
    label = config.get("experiment", {}).get("label")

    if label is None:
        # No experiment label - use original output path
        return config["simulation"]["output_path"]

    # Compute experiment hash
    exp_hash = compute_experiment_hash(config)

    # Use actual seed (never "noseed" - always the real seed used)
    seed_str = str(actual_seed)

    # Construct paths
    exp_dir = Path("experiments") / f"{label}-{exp_hash}"
    run_dir = exp_dir / seed_str
    output_path = run_dir / config["simulation"]["output_path"]

    # Create directories
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy config to experiment directory (if not already there)
    exp_config_path = exp_dir / "cfg.toml"
    if not exp_config_path.exists():
        shutil.copy2(config_file, exp_config_path)
        logger.info(f"Wrote experiment config to {exp_config_path}")

    # Write version info (git SHA) for reproducibility
    version_path = exp_dir / "version.txt"
    git_sha = get_git_sha()
    if not version_path.exists():
        with open(version_path, 'w') as f:
            f.write(f"git_sha={git_sha}\n")
        logger.info(f"Wrote version info to {version_path} (git_sha={git_sha[:7]})")

    logger.info(f"Experiment: {label}-{exp_hash}")
    logger.info(f"  Config hash: {exp_hash}")
    logger.info(f"  Seed: {seed_str}")
    logger.info(f"  Output: {output_path}")

    return str(output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_configuration(config: dict) -> None:
    """Print human-readable configuration summary."""
    sim = config.get("simulation", {})
    cat = config.get("catalog", {})
    txn = config.get("transaction", {})
    storage = config.get("storage", {})

    print("Configuration:")
    print(f"  Duration: {sim.get('duration_ms', 3600000)/1000:.0f}s")
    print(f"  Seed: {sim.get('seed', 'random')}")
    print(f"  Tables: {cat.get('num_tables', 1)}")
    print(f"  Provider: {storage.get('provider', 'instant')}")
    print(f"  Max retries: {txn.get('retry', 10)}")

    op_types = txn.get("operation_types", {})
    if op_types:
        parts = []
        for name in ["fast_append", "merge_append", "validated_overwrite"]:
            w = op_types.get(name, 0)
            if w > 0:
                parts.append(f"{name}={w:.0%}")
        if parts:
            print(f"  Operations: {', '.join(parts)}")


def confirm_run() -> bool:
    """Ask user to confirm simulation run."""
    try:
        response = input("\nProceed? [y/N] ").strip().lower()
        return response in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


def cli():
    """CLI entry point for endive simulator."""
    parser = argparse.ArgumentParser(
        description="Iceberg-style catalog simulator for exploring commit latency tradeoffs"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="cfg.toml",
        help="Path to TOML configuration file (default: cfg.toml)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all logging except errors"
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable DES engine profiling (writes .profile.json)"
    )
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Load and validate configuration
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Validate configuration
    validation_errors, validation_warnings = validate_config(config)
    if validation_warnings:
        print("Configuration warnings:")
        for warning in validation_warnings:
            print(f"  ! {warning}")
    if validation_errors:
        print("Configuration validation failed:")
        for error in validation_errors:
            print(f"  x {error}")
        sys.exit(1)

    # Check if experiment results already exist
    skip, existing_seed, existing_output = check_existing_experiment(config, args.config)
    if skip:
        if not args.quiet:
            print(f"Results already exist: {existing_output}")
            print(f"  Skipping simulation (seed={existing_seed})")
        sys.exit(0)

    # Print configuration (unless quiet mode)
    if not args.quiet:
        print_configuration(config)

    # Confirmation prompt (unless --yes or --quiet)
    if not args.yes and not args.quiet:
        if not confirm_run():
            print("Simulation cancelled.")
            sys.exit(0)

    # Setup random seed
    sim_cfg = config.get("simulation", {})
    seed = sim_cfg.get("seed")
    if seed is None:
        seed = np.random.randint(0, 2**32 - 1)
        logger.info(f"Using random seed: {seed}")
    else:
        logger.info(f"Using configured seed: {seed}")

    # Prepare experiment output directory with actual seed
    final_output_path = prepare_experiment_output(config, args.config, seed)

    # Load simulation configuration using new modules
    try:
        sim_config = load_simulation_config(args.config, seed_override=seed)
    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Run simulation with streaming export to temp file
    logger.info("Starting simulation...")
    output_dir = Path(final_output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_dir / ".running.parquet"

    progress_path = str(output_dir / ".progress.json")
    sim = Simulation(
        sim_config,
        output_path=str(temp_output_path),
        progress_path=progress_path,
        profile=getattr(args, 'profile', False),
    )
    stats = sim.run()

    logger.info("Simulation complete")

    # Rename to final output path on success
    logger.info(f"Moving results to {final_output_path}")
    shutil.move(str(temp_output_path), final_output_path)
    logger.info("Results exported successfully")

    # Print summary
    if not args.quiet:
        print(f"\nResults: {stats.committed} committed, {stats.aborted} aborted "
              f"({stats.total_retries} retries)")
        if stats.validation_exceptions > 0:
            print(f"  Validation exceptions: {stats.validation_exceptions}")
        print(f"  Output: {final_output_path}")


if __name__ == "__main__":
    cli()
