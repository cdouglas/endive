#!/usr/bin/env python3
"""
Remove experiment results that don't match the current session nonce.

This script:
1. Reads the current nonce from experiments/.nonce
2. For each experiment directory, regenerates expected seeds
3. Removes seed directories that don't match expected seeds
4. Removes experiment directories with no matching seeds

This cleans up duplicate results from previous runs with different nonces.
"""

import hashlib
import sys
import tomllib
import shutil
from pathlib import Path
from typing import Set, Tuple


def read_nonce() -> str:
    """Read the current session nonce."""
    nonce_file = Path("experiments/.nonce")
    if not nonce_file.exists():
        print("ERROR: No nonce file found at experiments/.nonce")
        print("Run the experiment script at least once to generate a nonce.")
        sys.exit(1)

    nonce = nonce_file.read_text().strip()
    print(f"Current nonce: {nonce}")
    return nonce


def generate_deterministic_seed(nonce: str, exp_label: str, param_string: str, seed_num: int) -> int:
    """Generate deterministic seed using same logic as bash script.

    Args:
        nonce: Session nonce
        exp_label: Experiment label (e.g., "exp3_1")
        param_string: Parameter string (e.g., "prob=0.5:load=500")
        seed_num: Seed number (1-5)

    Returns:
        Deterministic seed value
    """
    hash_input = f"{nonce}:{exp_label}:{param_string}:{seed_num}"
    hash_obj = hashlib.sha256(hash_input.encode('utf-8'))
    # Take first 8 hex digits (32 bits) and convert to int
    hex_str = hash_obj.hexdigest()[:8]
    seed = int(hex_str, 16)
    return seed


def extract_exp_label_and_params(exp_dir: Path, config: dict) -> Tuple[str, str]:
    """Extract experiment label and parameter string from config.

    Returns:
        (exp_label, param_string) tuple
    """
    # Get label from config
    label = config.get("experiment", {}).get("label", "")

    # Extract short label (e.g., "exp3_1" from "exp3_1_single_table_real")
    if "_" in label:
        parts = label.split("_")
        exp_label = f"{parts[0]}_{parts[1]}"  # e.g., "exp3_1"
    else:
        exp_label = label

    # Build parameter string based on experiment type
    # This needs to match what run_baseline_experiments.sh uses
    params = []

    if "exp2_1" in label or "exp4_1" in label:
        # Single table false conflicts: just load
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if load:
            params.append(f"load={int(load)}")

    elif "exp2_2" in label:
        # Multi-table false conflicts: tables and load
        num_tables = config.get("catalog", {}).get("num_tables")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if num_tables and load:
            params.append(f"tables={num_tables}")
            params.append(f"load={int(load)}")

    elif "exp3_1" in label:
        # Single table real conflicts: prob and load
        prob = config.get("transaction", {}).get("real_conflict_probability")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if prob is not None and load:
            params.append(f"prob={prob}")
            params.append(f"load={int(load)}")

    elif "exp3_2" in label:
        # Manifest distribution: dist and load
        dist = config.get("transaction", {}).get("conflicting_manifests", {}).get("distribution")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")

        if dist == "fixed":
            value = config.get("transaction", {}).get("conflicting_manifests", {}).get("value")
            if value and load:
                params.append(f"dist=fixed:{value}")
        else:  # exponential
            mean = config.get("transaction", {}).get("conflicting_manifests", {}).get("mean")
            if mean and load:
                params.append(f"dist=exponential:{mean}")

        if load:
            params.append(f"load={int(load)}")

    elif "exp3_3" in label or "exp3_4" in label:
        # Multi-table real conflicts: tables, prob, load
        num_tables = config.get("catalog", {}).get("num_tables")
        prob = config.get("transaction", {}).get("real_conflict_probability")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if num_tables and prob is not None and load:
            params.append(f"tables={num_tables}")
            params.append(f"prob={prob}")
            params.append(f"load={int(load)}")

    elif "exp5_1" in label:
        # Single table catalog latency: cas and load
        # T_CAS is under storage section
        cas = config.get("storage", {}).get("T_CAS", {}).get("mean")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if cas and load:
            params.append(f"cas={int(cas)}")
            params.append(f"load={int(load)}")

    elif "exp5_2" in label:
        # Multi-table catalog latency: cas, tables, load
        # T_CAS is under storage section
        cas = config.get("storage", {}).get("T_CAS", {}).get("mean")
        num_tables = config.get("catalog", {}).get("num_tables")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if cas and num_tables and load:
            params.append(f"cas={int(cas)}")
            params.append(f"tables={num_tables}")
            params.append(f"load={int(load)}")

    elif "exp5_3" in label:
        # Transaction partitioning catalog latency: cas, groups, load
        # T_CAS is under storage section
        cas = config.get("storage", {}).get("T_CAS", {}).get("mean")
        num_groups = config.get("catalog", {}).get("num_groups")
        load = config.get("transaction", {}).get("inter_arrival", {}).get("scale")
        if cas and num_groups and load:
            params.append(f"cas={int(cas)}")
            params.append(f"groups={num_groups}")
            params.append(f"load={int(load)}")

    param_string = ":".join(params)
    return exp_label, param_string


def get_expected_seeds(nonce: str, exp_label: str, param_string: str, num_seeds: int = 5) -> Set[int]:
    """Get the set of expected seeds for this experiment configuration."""
    expected = set()
    for seed_num in range(1, num_seeds + 1):
        seed = generate_deterministic_seed(nonce, exp_label, param_string, seed_num)
        expected.add(seed)
    return expected


def cleanup_experiment_dir(exp_dir: Path, nonce: str, dry_run: bool = False) -> Tuple[int, int]:
    """Clean up a single experiment directory.

    Returns:
        (removed_seeds, kept_seeds) tuple
    """
    # Read config
    cfg_file = exp_dir / "cfg.toml"
    if not cfg_file.exists():
        print(f"  WARNING: No cfg.toml in {exp_dir.name}, skipping")
        return (0, 0)

    try:
        with open(cfg_file, "rb") as f:
            config = tomllib.load(f)
    except Exception as e:
        print(f"  WARNING: Could not read config from {exp_dir.name}: {e}")
        return (0, 0)

    # Extract label and params
    exp_label, param_string = extract_exp_label_and_params(exp_dir, config)

    if not param_string:
        print(f"  WARNING: Could not extract parameters from {exp_dir.name}, skipping")
        return (0, 0)

    # Get expected seeds
    expected_seeds = get_expected_seeds(nonce, exp_label, param_string)

    # Check each seed directory
    removed = 0
    kept = 0

    for seed_dir in exp_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.isdigit():
            continue

        seed_value = int(seed_dir.name)

        if seed_value in expected_seeds:
            kept += 1
        else:
            removed += 1
            if dry_run:
                print(f"    [DRY RUN] Would remove: {seed_dir.name} (not in expected seeds)")
            else:
                print(f"    Removing: {seed_dir.name} (not in expected seeds)")
                shutil.rmtree(seed_dir)

    return (removed, kept)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove experiment results that don't match current nonce"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    parser.add_argument(
        "--experiments",
        default="experiments",
        help="Path to experiments directory (default: experiments)"
    )

    args = parser.parse_args()

    # Read nonce
    nonce = read_nonce()

    # Find all experiment directories
    exp_base = Path(args.experiments)
    if not exp_base.exists():
        print(f"ERROR: Experiments directory not found: {exp_base}")
        sys.exit(1)

    exp_dirs = sorted([d for d in exp_base.glob("exp*-*") if d.is_dir()])

    if not exp_dirs:
        print("No experiment directories found")
        return

    print(f"\nFound {len(exp_dirs)} experiment directories")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be removed ***\n")
    else:
        print("\n*** REMOVING MISMATCHED SEEDS ***\n")

    total_removed = 0
    total_kept = 0
    dirs_with_removals = []

    for exp_dir in exp_dirs:
        removed, kept = cleanup_experiment_dir(exp_dir, nonce, args.dry_run)

        if removed > 0:
            dirs_with_removals.append(exp_dir.name)
            print(f"  {exp_dir.name}: Removed {removed}, Kept {kept}")

        total_removed += removed
        total_kept += kept

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total seeds kept: {total_kept}")
    print(f"Total seeds removed: {total_removed}")
    print(f"Directories with removals: {len(dirs_with_removals)}")

    if dirs_with_removals and not args.dry_run:
        print(f"\nCleaned up {len(dirs_with_removals)} experiment directories")
    elif args.dry_run and total_removed > 0:
        print(f"\nRun without --dry-run to remove {total_removed} mismatched seeds")
    else:
        print("\nNo mismatched seeds found - all results match current nonce!")


if __name__ == "__main__":
    main()
