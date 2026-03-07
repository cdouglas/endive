#!/usr/bin/env python3
"""Experiment management utility for endive experiments.

Provides ExperimentStore for scanning, filtering, and managing experiments
across disk directories and consolidated parquet files.
"""
from __future__ import annotations

import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

import pyarrow.parquet as pq

# Add project root for imports from scripts/ and endive/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from endive.config import compute_code_hash, compute_experiment_hash


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ExperimentEntry:
    """One experiment hash (one unique config+code combination)."""

    label: str
    exp_hash: str
    group: str
    source: str  # "disk" | "consolidated" | "both"
    disk_dir: Path | None
    config: dict | None
    seeds_disk: set[int] = field(default_factory=set)
    seeds_consolidated: set[int] = field(default_factory=set)
    seeds_empty: set[int] = field(default_factory=set)
    seeds_active: set[int] = field(default_factory=set)
    seeds_crashed: set[int] = field(default_factory=set)
    disk_bytes: int = 0
    code_version: str = ""
    is_stale: bool = False

    @property
    def all_seeds(self) -> set[int]:
        return self.seeds_disk | self.seeds_consolidated

    @property
    def seed_count(self) -> int:
        return len(self.all_seeds)


# ---------------------------------------------------------------------------
# Label-to-group mapping
# ---------------------------------------------------------------------------

def _load_experiment_groups() -> dict[str, list[str]]:
    """Import EXPERIMENT_GROUPS from run_all_experiments.py."""
    from run_all_experiments import EXPERIMENT_GROUPS
    return EXPERIMENT_GROUPS


def _build_label_to_group(groups: dict[str, list[str]]) -> dict[str, str]:
    """Invert EXPERIMENT_GROUPS: config filename -> group name.

    Returns mapping from label (e.g. "exp1_fa_baseline") to group name.
    """
    label_to_group: dict[str, str] = {}
    for group_name, config_files in groups.items():
        for config_file in config_files:
            # "exp1_fa_baseline.toml" -> "exp1_fa_baseline"
            label = config_file.removesuffix(".toml")
            label_to_group[label] = group_name
    return label_to_group


def format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n < 1024:
        return f"{n} B"
    elif n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    elif n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    else:
        return f"{n / (1024 * 1024 * 1024):.1f} GB"


# ---------------------------------------------------------------------------
# ExperimentStore
# ---------------------------------------------------------------------------

class ExperimentStore:
    """Scans and indexes experiments from disk and consolidated parquet."""

    def __init__(
        self,
        base_dir: str | Path = "experiments",
        staleness_threshold_s: float = 30 * 60,
    ):
        self._base_dir = Path(base_dir)
        self._staleness_threshold_s = staleness_threshold_s
        self._entries: dict[tuple[str, str], ExperimentEntry] = {}  # (label, hash) -> entry
        self._label_to_group = _build_label_to_group(_load_experiment_groups())

    def scan(self) -> None:
        """Run full scan: disk directories, consolidated parquet, stale detection."""
        self._entries.clear()
        self._scan_disk()
        self._scan_consolidated()
        self._detect_stale_hashes()

    def _label_group(self, label: str) -> str:
        """Look up group for a label, falling back to 'unknown'."""
        return self._label_to_group.get(label, "unknown")

    # ----- disk scan -----

    def _scan_disk(self) -> None:
        """Iterate experiments/*-* directories and populate entries."""
        if not self._base_dir.is_dir():
            return

        now = time.time()

        for exp_dir in sorted(self._base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            dir_name = exp_dir.name
            # Must contain a hyphen separating label from hash
            if '-' not in dir_name:
                continue

            label, exp_hash = dir_name.rsplit('-', 1)
            # Hash must be >= 6 hex chars
            if len(exp_hash) < 6 or not all(c in '0123456789abcdef' for c in exp_hash):
                continue

            # Parse cfg.toml
            config = None
            cfg_path = exp_dir / "cfg.toml"
            if cfg_path.exists():
                try:
                    with open(cfg_path, "rb") as f:
                        config = tomllib.load(f)
                except Exception:
                    pass

            # Read version.txt for code_version
            code_version = ""
            version_path = exp_dir / "version.txt"
            if version_path.exists():
                try:
                    for line in version_path.read_text().splitlines():
                        if line.startswith("code_hash="):
                            code_version = line.split("=", 1)[1].strip()
                except Exception:
                    pass

            # Classify seed directories
            seeds_disk: set[int] = set()
            seeds_empty: set[int] = set()
            seeds_active: set[int] = set()
            seeds_crashed: set[int] = set()
            total_bytes = 0

            for child in exp_dir.iterdir():
                if not child.is_dir():
                    # Count file sizes for cfg.toml, version.txt etc
                    try:
                        total_bytes += child.stat().st_size
                    except OSError:
                        pass
                    continue

                # Try to parse as seed number
                try:
                    seed = int(child.name)
                except ValueError:
                    continue

                results_path = child / "results.parquet"
                running_path = child / ".running.parquet"
                progress_path = child / ".progress.json"

                if results_path.exists():
                    # Complete seed
                    seeds_disk.add(seed)
                    try:
                        total_bytes += results_path.stat().st_size
                    except OSError:
                        pass
                elif running_path.exists():
                    # In-progress or crashed
                    try:
                        total_bytes += running_path.stat().st_size
                    except OSError:
                        pass
                    if progress_path.exists():
                        try:
                            mtime = progress_path.stat().st_mtime
                            if (now - mtime) < self._staleness_threshold_s:
                                seeds_active.add(seed)
                            else:
                                seeds_crashed.add(seed)
                        except OSError:
                            seeds_crashed.add(seed)
                    else:
                        seeds_crashed.add(seed)
                else:
                    # Empty seed directory (interrupted before any output)
                    seeds_empty.add(seed)

            key = (label, exp_hash)
            entry = self._entries.get(key)
            if entry is None:
                entry = ExperimentEntry(
                    label=label,
                    exp_hash=exp_hash,
                    group=self._label_group(label),
                    source="disk",
                    disk_dir=exp_dir,
                    config=config,
                    code_version=code_version,
                )
                self._entries[key] = entry
            else:
                # Merge with existing entry (from consolidated)
                entry.source = "both"
                entry.disk_dir = exp_dir
                entry.config = entry.config or config
                entry.code_version = entry.code_version or code_version

            entry.seeds_disk = seeds_disk
            entry.seeds_empty = seeds_empty
            entry.seeds_active = seeds_active
            entry.seeds_crashed = seeds_crashed
            entry.disk_bytes = total_bytes

    # ----- consolidated scan -----

    def _scan_consolidated(self) -> None:
        """Read metadata from consolidated.parquet and per-label parquet files."""
        # Main consolidated file
        consolidated_path = self._base_dir / "consolidated.parquet"
        if consolidated_path.exists():
            self._scan_parquet_file(consolidated_path)

        # Per-label parquet files (from partitioned compaction)
        for pq_file in sorted(self._base_dir.glob("*.parquet")):
            if pq_file.name == "consolidated.parquet":
                continue
            self._scan_parquet_file(pq_file)

    def _scan_parquet_file(self, path: Path) -> None:
        """Extract (exp_name, exp_hash, seed) tuples from a parquet file.

        Reads one row per row group to avoid loading all 200M+ rows.
        Each row group corresponds to one seed of one experiment.
        """
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            return

        consolidated_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
        cols = ["exp_name", "exp_hash", "seed"]

        n_rg = pf.metadata.num_row_groups
        if n_rg > 100:
            # Large file (real consolidated): read one row per row group.
            # Each row group is one seed of one experiment.
            for i in range(n_rg):
                try:
                    rg = pf.read_row_group(i, columns=cols)
                    name = rg.column("exp_name")[0].as_py()
                    h = rg.column("exp_hash")[0].as_py()
                    s = int(rg.column("seed")[0].as_py())
                    consolidated_seeds[(name, h)].add(s)
                except Exception:
                    continue
        else:
            # Small file (tests, per-label): read all rows per row group.
            for i in range(n_rg):
                try:
                    rg = pf.read_row_group(i, columns=cols)
                    names = rg.column("exp_name").to_pylist()
                    hashes = rg.column("exp_hash").to_pylist()
                    rg_seeds = rg.column("seed").to_pylist()
                    for name, h, s in zip(names, hashes, rg_seeds):
                        consolidated_seeds[(name, h)].add(int(s))
                except Exception:
                    continue

        for (label, exp_hash), seed_set in consolidated_seeds.items():
            key = (label, exp_hash)
            entry = self._entries.get(key)
            if entry is None:
                entry = ExperimentEntry(
                    label=label,
                    exp_hash=exp_hash,
                    group=self._label_group(label),
                    source="consolidated",
                    disk_dir=None,
                    config=None,
                )
                self._entries[key] = entry
            else:
                if entry.source == "disk":
                    entry.source = "both"

            entry.seeds_consolidated = seed_set

    # ----- stale hash detection -----

    def _detect_stale_hashes(self) -> None:
        """Mark entries whose hash doesn't match current code + config."""
        current_code_hash = compute_code_hash()

        for entry in self._entries.values():
            if entry.config is None:
                # Can't verify without config — assume current if code matches
                if entry.code_version and entry.code_version != current_code_hash:
                    entry.is_stale = True
                continue

            # Recompute hash from stored config + current code
            try:
                expected_hash = compute_experiment_hash(entry.config)
            except Exception:
                # If hash computation fails, can't verify
                continue

            if expected_hash != entry.exp_hash:
                entry.is_stale = True

    # ----- filtering -----

    def get_entries(
        self,
        group: str | list[str] | None = None,
        pattern: str | None = None,
        exclude: str | None = None,
    ) -> list[ExperimentEntry]:
        """Filter entries by group name, fnmatch on label, or exclusion pattern.

        Args:
            group: Group name(s) to include. None = all groups.
            pattern: fnmatch pattern matched against label.
            exclude: fnmatch pattern to exclude by label.

        Returns:
            Filtered list of ExperimentEntry, sorted by (group, label, hash).
        """
        if isinstance(group, str):
            groups = [group]
        else:
            groups = group

        results = []
        for entry in self._entries.values():
            if groups is not None and entry.group not in groups:
                continue
            if pattern is not None and not fnmatch(entry.label, pattern):
                continue
            if exclude is not None and fnmatch(entry.label, exclude):
                continue
            results.append(entry)

        results.sort(key=lambda e: (e.group, e.label, e.exp_hash))
        return results

    @property
    def entries(self) -> list[ExperimentEntry]:
        """All entries, sorted by (group, label, hash)."""
        return self.get_entries()

    def labels(self) -> dict[str, list[ExperimentEntry]]:
        """Group entries by label. Returns {label: [entries]}."""
        by_label: dict[str, list[ExperimentEntry]] = defaultdict(list)
        for entry in self.entries:
            by_label[entry.label].append(entry)
        return dict(by_label)


# ---------------------------------------------------------------------------
# CLI entry point (placeholder for subcommands)
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="expctl",
        description="Experiment management utility",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # Placeholder subcommands — will be implemented in dependent issues
    sub_list = subparsers.add_parser("list", help="List experiments")
    sub_list.set_defaults(func=lambda _args: print("list: not yet implemented"))

    sub_compact = subparsers.add_parser("compact", help="Compact experiments")
    sub_compact.set_defaults(func=lambda _args: print("compact: not yet implemented"))

    sub_gc = subparsers.add_parser("gc", help="Garbage collect experiments")
    sub_gc.set_defaults(func=lambda _args: print("gc: not yet implemented"))

    sub_complete = subparsers.add_parser("complete", help="Complete missing seeds")
    sub_complete.set_defaults(func=lambda _args: print("complete: not yet implemented"))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
