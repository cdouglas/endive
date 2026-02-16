#!/usr/bin/env python3
"""
Run all blog post experiments with progress tracking and detach/reattach support.

Usage:
    # Run all experiments with default parallelism (4)
    python scripts/run_all_experiments.py

    # Run with specific parallelism
    python scripts/run_all_experiments.py --parallel 2

    # Run in background (detachable)
    nohup python scripts/run_all_experiments.py --parallel 4 > experiment_runner.log 2>&1 &

    # Check progress (while running or after detach)
    python scripts/run_all_experiments.py --status

    # Resume incomplete experiments
    python scripts/run_all_experiments.py --resume

    # Quick test mode (shorter duration, fewer seeds)
    python scripts/run_all_experiments.py --quick

    # Run specific experiment groups only
    python scripts/run_all_experiments.py --groups baseline,metadata

    # Dry run (show what would be executed)
    python scripts/run_all_experiments.py --dry-run
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ============================================================================
# Configuration
# ============================================================================

# Experiment groups and their base configs
EXPERIMENT_GROUPS = {
    # Blog post Questions 1a/1b/2a/2b (catalog conflicts)
    "trivial": [
        "single_table_trivial.toml",
        "single_table_trivial_backoff.toml",
    ],
    "mixed": [
        "single_table_mixed.toml",
    ],
    "multi_table": [
        "multi_table_trivial.toml",
        "multi_table_mixed.toml",
    ],
    # Blog post Question 3 (optimization impact)
    "baseline": [
        "baseline_s3.toml",
        "baseline_s3x.toml",
        "baseline_azure.toml",
        "baseline_azurex.toml",
    ],
    "metadata": [
        "metadata_s3.toml",
        "metadata_s3x.toml",
        "metadata_azure.toml",
        "metadata_azurex.toml",
    ],
    # NOTE: S3 Standard doesn't support conditional append, so excluded from ml_append/combined
    "ml_append": [
        "ml_append_s3x.toml",
        "ml_append_azure.toml",
        "ml_append_azurex.toml",
    ],
    "combined": [
        "combined_optimizations_s3x.toml",
        "combined_optimizations_azure.toml",
        "combined_optimizations_azurex.toml",
    ],
    # Instant catalog experiments (1ms CAS, real S3 storage)
    # Shows table metadata is the bottleneck, not catalog
    "instant_trivial": [
        "instant_1tbl_trivial.toml",
        "instant_ntbl_trivial.toml",
    ],
    "instant_nontrivial": [
        "instant_1tbl_nontrivial.toml",
        "instant_ntbl_nontrivial.toml",
    ],
    # Partition scaling experiments
    # Shows how partitioning breaks single-table bottleneck
    "partition_scaling": [
        "instant_partition_scaling.toml",
    ],
    "partition_vs_tables": [
        "instant_partition_vs_tables.toml",
    ],
}

# Parameter sweeps
# Note: scale=10ms excluded - system saturates completely and can't make progress
LOAD_SWEEP = [20, 50, 100, 200, 500, 1000, 2000, 5000]
CONFLICT_PROB_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
NUM_TABLES_SWEEP = [1, 2, 5, 10, 20, 50]
NUM_TABLES_MIXED_SWEEP = [2, 5, 10, 20]
CONFLICT_PROB_MIXED_SWEEP = [0.0, 0.1, 0.3, 0.5]
NUM_PARTITIONS_SWEEP = [1, 5, 10, 25, 50, 100]
NUM_PARTITIONS_COMPARE_SWEEP = [2, 5, 10, 20]  # Match num_tables for comparison

# Quick mode parameters
QUICK_LOADS = [100, 500, 2000]
QUICK_DURATION = 60000  # 1 minute

# Paths
CONFIG_DIR = Path("experiment_configs")
EXPERIMENTS_DIR = Path("experiments")
STATE_FILE = EXPERIMENTS_DIR / ".runner_state.json"

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExperimentRun:
    """Single experiment run configuration."""
    config_path: str
    label: str
    seed: int
    params: dict = field(default_factory=dict)

    @property
    def param_str(self) -> str:
        return "_".join(f"{k}={v}" for k, v in sorted(self.params.items()))

    @property
    def run_id(self) -> str:
        """Unique identifier for this run."""
        return f"{self.label}_{self.param_str}_s{self.seed}"

@dataclass
class RunnerState:
    """Persistent state for resume capability."""
    started_at: str
    total_runs: int
    completed: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    in_progress: list = field(default_factory=list)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump({
                'started_at': self.started_at,
                'total_runs': self.total_runs,
                'completed': self.completed,
                'failed': self.failed,
                'in_progress': self.in_progress,
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional['RunnerState']:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

# ============================================================================
# Helper Functions
# ============================================================================

def generate_seed(nonce: str, label: str, params: dict, seed_num: int) -> int:
    """Generate deterministic seed from parameters."""
    param_str = ":".join(f"{k}={v}" for k, v in sorted(params.items()))
    hash_input = f"{nonce}:{label}:{param_str}:{seed_num}"
    hash_bytes = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    return int(hash_bytes, 16)

def get_or_create_nonce() -> str:
    """Get or create session nonce for deterministic seeds."""
    nonce_file = EXPERIMENTS_DIR / ".nonce"
    EXPERIMENTS_DIR.mkdir(exist_ok=True)

    if nonce_file.exists():
        return nonce_file.read_text().strip()

    nonce = hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]
    nonce_file.write_text(nonce)
    return nonce

def create_config_variant(base_config: Path, params: dict, seed: int,
                          duration_ms: Optional[int] = None) -> Path:
    """Create a temporary config file with modified parameters."""
    content = base_config.read_text()

    # Add seed after [simulation]
    if "[simulation]" in content:
        content = content.replace("[simulation]", f"[simulation]\nseed = {seed}")

    # Override duration if specified
    if duration_ms:
        import re
        content = re.sub(r'duration_ms\s*=\s*\d+', f'duration_ms = {duration_ms}', content)

    # Apply parameter overrides
    import re
    for key, value in params.items():
        # Escape key for regex (handles dots in key names like inter_arrival.scale)
        escaped_key = re.escape(key)
        pattern = rf'^{escaped_key}\s*=\s*.*$'
        replacement = f'{key} = {value}'
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Write to temp file
    fd, path = tempfile.mkstemp(suffix='.toml')
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)

def run_single_experiment(run: ExperimentRun, duration_ms: Optional[int] = None) -> tuple:
    """Run a single experiment. Returns (run_id, success, message)."""
    try:
        base_config = CONFIG_DIR / run.config_path
        if not base_config.exists():
            return (run.run_id, False, f"Config not found: {base_config}")

        # Create variant config
        config_path = create_config_variant(base_config, run.params, run.seed, duration_ms)

        try:
            # Run simulation
            result = subprocess.run(
                ["python", "-m", "endive.main", str(config_path), "--yes"],
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )

            success = result.returncode == 0
            message = "OK" if success else result.stderr[-500:] if result.stderr else "Unknown error"
            return (run.run_id, success, message)

        finally:
            # Clean up temp config
            config_path.unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        return (run.run_id, False, "Timeout after 2 hours")
    except Exception as e:
        return (run.run_id, False, str(e))

def check_experiment_exists(run: ExperimentRun) -> bool:
    """Check if experiment results already exist."""
    # Look for experiment directories matching this run
    pattern = f"{run.label}-*"
    for exp_dir in EXPERIMENTS_DIR.glob(pattern):
        seed_dir = exp_dir / str(run.seed)
        if (seed_dir / "results.parquet").exists():
            return True
    return False

# ============================================================================
# Experiment Generation
# ============================================================================

def generate_all_runs(groups: list, num_seeds: int, quick: bool = False) -> list:
    """Generate all experiment runs."""
    runs = []
    nonce = get_or_create_nonce()

    loads = QUICK_LOADS if quick else LOAD_SWEEP

    for group in groups:
        if group not in EXPERIMENT_GROUPS:
            print(f"Warning: Unknown group '{group}', skipping")
            continue

        configs = EXPERIMENT_GROUPS[group]

        for config_name in configs:
            base_name = config_name.replace(".toml", "")

            # Determine sweep parameters based on config type
            if "single_table_trivial" in config_name:
                # Sweep load only
                for load in loads:
                    for seed_num in range(1, num_seeds + 1):
                        seed = generate_seed(nonce, base_name, {"load": load}, seed_num)
                        runs.append(ExperimentRun(
                            config_path=config_name,
                            label=base_name,
                            seed=seed,
                            params={"inter_arrival.scale": float(load)}
                        ))

            elif "single_table_mixed" in config_name:
                # Sweep conflict probability at fixed load
                probs = [0.0, 0.3, 0.7, 1.0] if quick else CONFLICT_PROB_SWEEP
                for prob in probs:
                    for seed_num in range(1, num_seeds + 1):
                        seed = generate_seed(nonce, base_name, {"prob": prob}, seed_num)
                        runs.append(ExperimentRun(
                            config_path=config_name,
                            label=base_name,
                            seed=seed,
                            params={"real_conflict_probability": prob}
                        ))

            elif "multi_table_trivial" in config_name:
                # Sweep num_tables at fixed load
                tables = [1, 5, 20] if quick else NUM_TABLES_SWEEP
                for nt in tables:
                    for seed_num in range(1, num_seeds + 1):
                        seed = generate_seed(nonce, base_name, {"tables": nt}, seed_num)
                        runs.append(ExperimentRun(
                            config_path=config_name,
                            label=base_name,
                            seed=seed,
                            params={"num_tables": nt}
                        ))

            elif "multi_table_mixed" in config_name:
                # Sweep num_tables x conflict_probability
                tables = [2, 10] if quick else NUM_TABLES_MIXED_SWEEP
                probs = [0.0, 0.3] if quick else CONFLICT_PROB_MIXED_SWEEP
                for nt in tables:
                    for prob in probs:
                        for seed_num in range(1, num_seeds + 1):
                            seed = generate_seed(nonce, base_name, {"tables": nt, "prob": prob}, seed_num)
                            runs.append(ExperimentRun(
                                config_path=config_name,
                                label=base_name,
                                seed=seed,
                                params={"num_tables": nt, "real_conflict_probability": prob}
                            ))

            elif "instant_1tbl_trivial" in config_name:
                # Instant catalog, single table, trivial: sweep load only
                for load in loads:
                    for seed_num in range(1, num_seeds + 1):
                        seed = generate_seed(nonce, base_name, {"load": load}, seed_num)
                        runs.append(ExperimentRun(
                            config_path=config_name,
                            label=base_name,
                            seed=seed,
                            params={"inter_arrival.scale": float(load)}
                        ))

            elif "instant_1tbl_nontrivial" in config_name:
                # Instant catalog, single table, non-trivial: sweep load x prob
                probs = [0.0, 0.3, 1.0] if quick else [0.0, 0.1, 0.3, 0.5, 1.0]
                for load in loads:
                    for prob in probs:
                        for seed_num in range(1, num_seeds + 1):
                            seed = generate_seed(nonce, base_name, {"load": load, "prob": prob}, seed_num)
                            runs.append(ExperimentRun(
                                config_path=config_name,
                                label=base_name,
                                seed=seed,
                                params={"inter_arrival.scale": float(load), "real_conflict_probability": prob}
                            ))

            elif "instant_ntbl_trivial" in config_name:
                # Instant catalog, multi-table, trivial: sweep load x tables
                tables = [1, 5, 20] if quick else [1, 2, 5, 10, 20]
                for load in loads:
                    for nt in tables:
                        for seed_num in range(1, num_seeds + 1):
                            seed = generate_seed(nonce, base_name, {"load": load, "tables": nt}, seed_num)
                            runs.append(ExperimentRun(
                                config_path=config_name,
                                label=base_name,
                                seed=seed,
                                params={"inter_arrival.scale": float(load), "num_tables": nt}
                            ))

            elif "instant_ntbl_nontrivial" in config_name:
                # Instant catalog, multi-table, non-trivial: sweep load x tables x prob
                tables = [2, 10] if quick else [2, 5, 10]
                probs = [0.0, 0.5] if quick else [0.0, 0.3, 0.5]
                for load in loads:
                    for nt in tables:
                        for prob in probs:
                            for seed_num in range(1, num_seeds + 1):
                                seed = generate_seed(nonce, base_name, {"load": load, "tables": nt, "prob": prob}, seed_num)
                                runs.append(ExperimentRun(
                                    config_path=config_name,
                                    label=base_name,
                                    seed=seed,
                                    params={"inter_arrival.scale": float(load), "num_tables": nt, "real_conflict_probability": prob}
                                ))

            elif "instant_partition_scaling" in config_name:
                # Partition scaling: sweep load x num_partitions
                partitions = [1, 10, 100] if quick else NUM_PARTITIONS_SWEEP
                for load in loads:
                    for np in partitions:
                        for seed_num in range(1, num_seeds + 1):
                            seed = generate_seed(nonce, base_name, {"load": load, "partitions": np}, seed_num)
                            runs.append(ExperimentRun(
                                config_path=config_name,
                                label=base_name,
                                seed=seed,
                                params={"inter_arrival.scale": float(load), "partition.num_partitions": np}
                            ))

            elif "instant_partition_vs_tables" in config_name:
                # Partition vs tables comparison: sweep load x num_partitions (matching table counts)
                partitions = [2, 10] if quick else NUM_PARTITIONS_COMPARE_SWEEP
                for load in loads:
                    for np in partitions:
                        for seed_num in range(1, num_seeds + 1):
                            seed = generate_seed(nonce, base_name, {"load": load, "partitions": np}, seed_num)
                            runs.append(ExperimentRun(
                                config_path=config_name,
                                label=base_name,
                                seed=seed,
                                params={"inter_arrival.scale": float(load), "partition.num_partitions": np}
                            ))

            else:
                # Optimization experiments: sweep load only
                for load in loads:
                    for seed_num in range(1, num_seeds + 1):
                        seed = generate_seed(nonce, base_name, {"load": load}, seed_num)
                        runs.append(ExperimentRun(
                            config_path=config_name,
                            label=base_name,
                            seed=seed,
                            params={"inter_arrival.scale": float(load)}
                        ))

    return runs

# ============================================================================
# Main Runner
# ============================================================================

def print_status():
    """Print current runner status."""
    state = RunnerState.load(STATE_FILE)
    if not state:
        print("No runner state found. Start experiments with: python scripts/run_all_experiments.py")
        return

    completed = len(state.completed)
    failed = len(state.failed)
    in_progress = len(state.in_progress)
    remaining = state.total_runs - completed - failed

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT RUNNER STATUS")
    print(f"{'='*60}")
    print(f"  Started:     {state.started_at}")
    print(f"  Total runs:  {state.total_runs}")
    print(f"  Completed:   {completed} ({100*completed/state.total_runs:.1f}%)")
    print(f"  Failed:      {failed}")
    print(f"  In progress: {in_progress}")
    print(f"  Remaining:   {remaining}")
    print(f"{'='*60}\n")

    if failed:
        print("Failed runs:")
        for run_id in state.failed[:5]:
            print(f"  - {run_id}")
        if len(state.failed) > 5:
            print(f"  ... and {len(state.failed) - 5} more")

def run_experiments(args):
    """Main experiment runner."""
    # Parse groups
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",")]
    else:
        groups = list(EXPERIMENT_GROUPS.keys())

    print(f"\n{'='*60}")
    print(f"  ENDIVE EXPERIMENT RUNNER")
    print(f"{'='*60}")
    print(f"  Groups:      {', '.join(groups)}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Parallel:    {args.parallel}")
    print(f"  Quick mode:  {args.quick}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'='*60}\n")

    # Generate all runs
    all_runs = generate_all_runs(groups, args.seeds, args.quick)

    # Filter out completed runs if resuming
    if args.resume:
        state = RunnerState.load(STATE_FILE)
        if state:
            completed_ids = set(state.completed)
            all_runs = [r for r in all_runs if r.run_id not in completed_ids]
            print(f"Resuming: {len(all_runs)} runs remaining")

    # Skip existing experiments unless forced
    if not args.force:
        original_count = len(all_runs)
        all_runs = [r for r in all_runs if not check_experiment_exists(r)]
        skipped = original_count - len(all_runs)
        if skipped > 0:
            print(f"Skipping {skipped} experiments with existing results")

    if not all_runs:
        print("No experiments to run!")
        return

    print(f"Total experiments to run: {len(all_runs)}")

    # Estimate time
    avg_time = 4 if args.quick else 180  # seconds per experiment
    total_time = (len(all_runs) * avg_time) / args.parallel
    est_time = timedelta(seconds=int(total_time))
    print(f"Estimated time: {est_time} (with {args.parallel} parallel workers)")

    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        for run in all_runs[:10]:
            print(f"  {run.run_id}")
        if len(all_runs) > 10:
            print(f"  ... and {len(all_runs) - 10} more")
        return

    # Initialize state
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    state = RunnerState(
        started_at=datetime.now().isoformat(),
        total_runs=len(all_runs),
        completed=[],
        failed=[],
        in_progress=[]
    )
    state.save(STATE_FILE)

    # Duration override for quick mode
    duration_ms = QUICK_DURATION if args.quick else None

    # Progress tracking
    completed = 0
    failed = 0
    start_time = time.time()

    # Run experiments
    if HAS_TQDM:
        pbar = tqdm(total=len(all_runs), desc="Running experiments", unit="exp")

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_single_experiment, run, duration_ms): run
            for run in all_runs
        }

        for future in as_completed(futures):
            run = futures[future]
            run_id, success, message = future.result()

            if success:
                completed += 1
                state.completed.append(run_id)
            else:
                failed += 1
                state.failed.append(run_id)
                if not HAS_TQDM:
                    print(f"  FAILED: {run_id}: {message[:100]}")

            # Update state file periodically
            state.save(STATE_FILE)

            if HAS_TQDM:
                pbar.update(1)
                pbar.set_postfix(ok=completed, fail=failed)
            else:
                elapsed = time.time() - start_time
                pct = (completed + failed) / len(all_runs) * 100
                print(f"[{pct:5.1f}%] {completed + failed}/{len(all_runs)} "
                      f"(ok={completed}, fail={failed}, elapsed={elapsed:.0f}s)")

    if HAS_TQDM:
        pbar.close()

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:  {timedelta(seconds=int(elapsed))}")
    print(f"  Completed:   {completed}")
    print(f"  Failed:      {failed}")
    print(f"{'='*60}")

    if failed > 0:
        print("\nFailed experiments:")
        for run_id in state.failed[:10]:
            print(f"  - {run_id}")
        if len(state.failed) > 10:
            print(f"  ... and {len(state.failed) - 10} more")
        print("\nRe-run failed experiments with: python scripts/run_all_experiments.py --resume")

# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run all blog post experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--parallel", "-p", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--seeds", "-s", type=int, default=3,
                        help="Number of seeds per configuration (default: 3)")
    parser.add_argument("--groups", "-g", type=str, default=None,
                        help="Comma-separated experiment groups to run "
                             "(default: all). Options: " +
                             ", ".join(EXPERIMENT_GROUPS.keys()))
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick test mode (1 min duration, fewer params)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be run without executing")
    parser.add_argument("--status", action="store_true",
                        help="Show current runner status")
    parser.add_argument("--resume", "-r", action="store_true",
                        help="Resume from previous state, skipping completed")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force re-run of existing experiments")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    run_experiments(args)

if __name__ == "__main__":
    main()
