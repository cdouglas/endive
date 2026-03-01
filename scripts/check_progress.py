#!/usr/bin/env python3
"""Check progress of running experiments.

Reads .runner_state.json and per-simulation .progress.json files
to provide an aggregate view of experiment progress.

Usage:
    python scripts/check_progress.py
    python scripts/check_progress.py --watch
    python scripts/check_progress.py --watch --interval 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXPERIMENTS_DIR = Path("experiments")
STATE_FILE = EXPERIMENTS_DIR / ".runner_state.json"


def load_runner_state() -> dict | None:
    """Load runner state from JSON file."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def find_progress_files() -> list[dict]:
    """Find and load all .progress.json files from running simulations."""
    progress = []
    if not EXPERIMENTS_DIR.exists():
        return progress
    for p in EXPERIMENTS_DIR.glob("*/*/.progress.json"):
        try:
            with open(p) as f:
                data = json.load(f)
            data["_path"] = str(p)
            progress.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return progress


def print_progress(clear: bool = False):
    """Print aggregate progress summary."""
    if clear:
        # ANSI escape to clear screen and move cursor to top
        print("\033[2J\033[H", end="")

    state = load_runner_state()
    progress_files = find_progress_files()

    print(f"{'=' * 60}")
    print(f"  EXPERIMENT PROGRESS")
    print(f"{'=' * 60}")

    if state:
        total = state.get("total_runs", 0)
        completed = len(state.get("completed", []))
        failed = len(state.get("failed", []))
        in_progress = len(state.get("in_progress", []))
        remaining = total - completed - failed

        print(f"  Started:     {state.get('started_at', 'unknown')}")
        print(f"  Total:       {total} experiments")
        pct = 100 * completed / total if total > 0 else 0
        print(f"  Completed:   {completed} ({pct:.1f}%)")
        print(f"  Failed:      {failed}")
        print(f"  In progress: {in_progress}")
        print(f"  Remaining:   {remaining}")
    else:
        print("  No runner state found.")
        print("  Start experiments with: python scripts/run_all_experiments.py")

    print(f"{'=' * 60}")

    if progress_files:
        avg_pct = sum(p.get("pct", 0) for p in progress_files) / len(progress_files)
        avg_wall = sum(p.get("wall_clock_s", 0) for p in progress_files) / len(progress_files)
        total_committed = sum(p.get("committed", 0) for p in progress_files)
        total_aborted = sum(p.get("aborted", 0) for p in progress_files)

        print(f"\n  Active Simulations ({len(progress_files)}):")
        print(f"    Avg progress:    {avg_pct:.1f}% of sim time")
        print(f"    Avg wall time:   {avg_wall:.1f}s")
        print(f"    Total committed: {total_committed}")
        print(f"    Total aborted:   {total_aborted}")

        # DES engine stats (if available in progress files)
        sims_with_des = [p for p in progress_files if "des_rate" in p]
        if sims_with_des:
            avg_des_rate = sum(p["des_rate"] for p in sims_with_des) / len(sims_with_des)
            max_queue = max(p.get("queue_depth", 0) for p in sims_with_des)
            print(f"    Avg DES rate:    {avg_des_rate:.0f} events/s")
            print(f"    Max queue depth: {max_queue}")

        # Estimate remaining time from wall-clock rate
        # Average: how many wall seconds per sim-percent
        sims_with_rate = [
            p for p in progress_files
            if p.get("pct", 0) > 0 and p.get("wall_clock_s", 0) > 0
        ]
        if sims_with_rate and state:
            avg_s_per_pct = sum(
                p["wall_clock_s"] / p["pct"] for p in sims_with_rate
            ) / len(sims_with_rate)
            avg_remaining_s = avg_s_per_pct * (100 - avg_pct)

            remaining_exps = state.get("total_runs", 0) - len(state.get("completed", [])) - len(state.get("failed", []))
            in_flight = len(progress_files)
            parallel = max(in_flight, 1)
            # Current batch finishes in avg_remaining_s
            # Remaining batches after this
            remaining_after_batch = max(0, remaining_exps - in_flight)
            full_sim_s = avg_s_per_pct * 100
            batches_left = remaining_after_batch / parallel if parallel > 0 else 0
            eta_s = avg_remaining_s + batches_left * full_sim_s

            if eta_s < 120:
                eta_str = f"{eta_s:.0f} seconds"
            elif eta_s < 7200:
                eta_str = f"{eta_s / 60:.1f} minutes"
            else:
                eta_str = f"{eta_s / 3600:.1f} hours"
            print(f"\n  Overall ETA: ~{eta_str} remaining")

    elif state and len(state.get("completed", [])) < state.get("total_runs", 0):
        print("\n  No active simulations found (may be between batches)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Check experiment progress")
    parser.add_argument(
        "--watch", "-w", action="store_true",
        help="Continuously refresh progress display",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=10,
        help="Refresh interval in seconds for --watch mode (default: 10)",
    )
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                print_progress(clear=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_progress()


if __name__ == "__main__":
    main()
