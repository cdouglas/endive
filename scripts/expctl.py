#!/usr/bin/env python3
"""Experiment management utility for endive experiments.

Provides ExperimentStore for scanning, filtering, and managing experiments
across disk directories and consolidated parquet files.
"""
from __future__ import annotations

import json
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
    stale_reasons: set[str] = field(default_factory=set)

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
        """Mark entries whose hash doesn't match current code, stored config,
        or source template in experiment_configs/."""
        from endive.config import compute_template_hash

        current_code_hash = compute_code_hash()
        template_hash_cache: dict[str, str | None] = {}

        def _live_template_hash(template_path: str) -> str | None:
            if template_path in template_hash_cache:
                return template_hash_cache[template_path]
            try:
                h = compute_template_hash(_PROJECT_ROOT / template_path)
            except (OSError, ValueError):
                h = None
            template_hash_cache[template_path] = h
            return h

        for entry in self._entries.values():
            if entry.config is None:
                # Can't verify without config — assume current if code matches
                if entry.code_version and entry.code_version != current_code_hash:
                    entry.is_stale = True
                    entry.stale_reasons.add("code")
                continue

            # 1. Recompute variant hash from stored config + current code
            try:
                expected_hash = compute_experiment_hash(entry.config)
            except Exception:
                expected_hash = None

            if expected_hash is not None and expected_hash != entry.exp_hash:
                entry.is_stale = True
                entry.stale_reasons.add("code-or-config")

            # 2. Check template drift: stored template_hash vs live source template.
            exp_section = entry.config.get("experiment", {}) or {}
            stored_template_hash = exp_section.get("template_hash")
            stored_template_path = exp_section.get("template_path")
            if stored_template_hash and stored_template_path:
                live = _live_template_hash(stored_template_path)
                if live is None:
                    entry.is_stale = True
                    entry.stale_reasons.add("template-missing")
                elif live != stored_template_hash:
                    entry.is_stale = True
                    entry.stale_reasons.add("template")

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

        results.sort(key=lambda e: (e.label, e.exp_hash))
        return results

    @property
    def entries(self) -> list[ExperimentEntry]:
        """All entries, sorted by (label, hash)."""
        return self.get_entries()

    def labels(self) -> dict[str, list[ExperimentEntry]]:
        """Group entries by label. Returns {label: [entries]}."""
        by_label: dict[str, list[ExperimentEntry]] = defaultdict(list)
        for entry in self.entries:
            by_label[entry.label].append(entry)
        return dict(by_label)


# ---------------------------------------------------------------------------
# list subcommand
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    """List experiments with optional filtering and output modes."""
    store = ExperimentStore(base_dir=args.dir)
    store.scan()

    entries = store.get_entries(
        group=args.group if args.group else None,
        pattern=args.pattern,
        exclude=args.exclude,
    )
    if args.stale:
        entries = [e for e in entries if e.is_stale]

    if args.json:
        _list_json(entries)
    elif args.physical:
        _list_physical(entries)
    elif args.verbose:
        _list_verbose(entries)
    else:
        _list_default(entries)


def _synthesize_status(label_entries: list[ExperimentEntry]) -> str:
    """Synthesize status flags from entries sharing a label.

    Multiple flags can appear (e.g. "stale (a1b2c3), incomplete").
    """
    flags: list[str] = []
    if any(e.is_stale for e in label_entries):
        reasons: set[str] = set()
        for e in label_entries:
            if e.is_stale:
                reasons.update(e.stale_reasons)
        if reasons:
            flags.append(f"stale ({', '.join(sorted(reasons))})")
        else:
            flags.append("stale")
    if any(e.seeds_active for e in label_entries):
        flags.append("active")
    if any(e.seeds_crashed for e in label_entries):
        flags.append("crashed")
    max_seeds = max(e.seed_count for e in label_entries) if label_entries else 0
    if max_seeds > 0 and any(e.seed_count < max_seeds for e in label_entries):
        flags.append("incomplete")
    return ", ".join(flags)


def _aggregate_source(label_entries: list[ExperimentEntry]) -> str:
    """Aggregate source across entries for a label."""
    sources = {e.source for e in label_entries}
    if "both" in sources:
        return "both"
    if len(sources) > 1:
        return "both"
    return sources.pop() if sources else "disk"


def _list_default(entries: list[ExperimentEntry]) -> None:
    """Per-label aggregation table."""
    if not entries:
        print("No experiments found.")
        return

    # Group by label
    by_label: dict[str, list[ExperimentEntry]] = defaultdict(list)
    for e in entries:
        by_label[e.label].append(e)

    # Build rows
    rows: list[tuple[str, str, int, str, str, str, str]] = []
    total_seeds = 0
    total_configs = 0
    total_bytes = 0

    for label in sorted(by_label):
        label_entries = by_label[label]
        group = label_entries[0].group
        n_configs = len(label_entries)
        all_seeds: set[int] = set()
        for e in label_entries:
            all_seeds |= e.all_seeds
        max_per_hash = max(e.seed_count for e in label_entries)
        seeds_str = f"{len(all_seeds)}/{max_per_hash}"
        source = _aggregate_source(label_entries)
        disk_bytes = sum(e.disk_bytes for e in label_entries)
        size_str = format_bytes(disk_bytes) if disk_bytes > 0 else "\u2014"
        status = _synthesize_status(label_entries)

        rows.append((group, label, n_configs, seeds_str, source, size_str, status))
        total_seeds += len(all_seeds)
        total_configs += n_configs
        total_bytes += disk_bytes

    # Compute column widths
    headers = ("Group", "Label", "Configs", "Seeds", "Source", "Size", "Status")
    w_group = max(len(headers[0]), max(len(r[0]) for r in rows))
    w_label = max(len(headers[1]), max(len(r[1]) for r in rows))
    w_configs = max(len(headers[2]), max(len(str(r[2])) for r in rows))
    w_seeds = max(len(headers[3]), max(len(r[3]) for r in rows))
    w_source = max(len(headers[4]), max(len(r[4]) for r in rows))
    w_size = max(len(headers[5]), max(len(r[5]) for r in rows))

    # Print header
    hdr = (
        f"{headers[0]:<{w_group}}  {headers[1]:<{w_label}}  "
        f"{headers[2]:>{w_configs}}  {headers[3]:>{w_seeds}}  "
        f"{headers[4]:<{w_source}}  {headers[5]:>{w_size}}  {headers[6]}"
    )
    print()
    print(hdr)

    # Print rows
    for group, label, n_configs, seeds_str, source, size_str, status in rows:
        line = (
            f"{group:<{w_group}}  {label:<{w_label}}  "
            f"{n_configs:>{w_configs}}  {seeds_str:>{w_seeds}}  "
            f"{source:<{w_source}}  {size_str:>{w_size}}  {status}"
        )
        print(line)

    # Summary
    n_labels = len(by_label)
    print(f"\n{n_labels} labels, {total_configs} configs, "
          f"{total_seeds} seeds ({format_bytes(total_bytes)} on disk)")


def _list_verbose(entries: list[ExperimentEntry]) -> None:
    """Per-hash detail table."""
    if not entries:
        print("No experiments found.")
        return

    # Build rows
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    total_seeds = 0
    total_bytes = 0

    for e in entries:
        nd = len(e.seeds_disk)
        nc = len(e.seeds_consolidated)
        nt = e.seed_count
        seeds_str = f"{nd}d + {nc}c ({nt})"
        if e.seeds_active:
            seeds_str += f"  +{len(e.seeds_active)} active"
        if e.seeds_crashed:
            seeds_str += f"  +{len(e.seeds_crashed)} crashed"
        size_str = format_bytes(e.disk_bytes) if e.disk_bytes > 0 else "\u2014"
        if e.is_stale and e.code_version:
            status = f"stale ({e.code_version[:7]})"
        elif e.is_stale:
            status = "stale"
        else:
            status = ""
        rows.append((e.group, e.label, e.exp_hash[:8], seeds_str, e.source, size_str, status))
        total_seeds += nt
        total_bytes += e.disk_bytes

    headers = ("Group", "Label", "Hash", "Seeds", "Source", "Size", "Status")
    w_group = max(len(headers[0]), max(len(r[0]) for r in rows))
    w_label = max(len(headers[1]), max(len(r[1]) for r in rows))
    w_hash = max(len(headers[2]), 8)
    w_seeds = max(len(headers[3]), max(len(r[3]) for r in rows))
    w_source = max(len(headers[4]), max(len(r[4]) for r in rows))
    w_size = max(len(headers[5]), max(len(r[5]) for r in rows))

    hdr = (
        f"{headers[0]:<{w_group}}  {headers[1]:<{w_label}}  "
        f"{headers[2]:<{w_hash}}  {headers[3]:<{w_seeds}}  "
        f"{headers[4]:<{w_source}}  {headers[5]:>{w_size}}  {headers[6]}"
    )
    print()
    print(hdr)

    for group, label, h, seeds_str, source, size_str, status in rows:
        line = (
            f"{group:<{w_group}}  {label:<{w_label}}  "
            f"{h:<{w_hash}}  {seeds_str:<{w_seeds}}  "
            f"{source:<{w_source}}  {size_str:>{w_size}}  {status}"
        )
        print(line)

    n_configs = len(entries)
    print(f"\n{n_configs} configs, {total_seeds} seeds ({format_bytes(total_bytes)} on disk)")


def _list_physical(entries: list[ExperimentEntry]) -> None:
    """Per-directory rows (disk-only)."""
    disk_entries = [e for e in entries if e.disk_dir is not None]
    if not disk_entries:
        print("No disk directories found.")
        return

    rows: list[tuple[str, int, int, int, int, str, str]] = []
    total_bytes = 0

    for e in disk_entries:
        dir_name = e.disk_dir.name
        n_seeds = len(e.seeds_disk)
        n_active = len(e.seeds_active)
        n_crashed = len(e.seeds_crashed)
        n_empty = len(e.seeds_empty)
        size_str = format_bytes(e.disk_bytes) if e.disk_bytes > 0 else "\u2014"
        if e.is_stale and e.code_version:
            status = f"stale ({e.code_version[:7]})"
        elif e.is_stale:
            status = "stale"
        else:
            status = ""
        rows.append((dir_name, n_seeds, n_active, n_crashed, n_empty, size_str, status))
        total_bytes += e.disk_bytes

    headers = ("Directory", "Seeds", "Active", "Crashed", "Empty", "Size", "Status")
    w_dir = max(len(headers[0]), max(len(r[0]) for r in rows))
    w_seeds = max(len(headers[1]), max(len(str(r[1])) for r in rows))
    w_active = max(len(headers[2]), max(len(str(r[2])) for r in rows))
    w_crashed = max(len(headers[3]), max(len(str(r[3])) for r in rows))
    w_empty = max(len(headers[4]), max(len(str(r[4])) for r in rows))
    w_size = max(len(headers[5]), max(len(r[5]) for r in rows))

    hdr = (
        f"{headers[0]:<{w_dir}}  {headers[1]:>{w_seeds}}  "
        f"{headers[2]:>{w_active}}  {headers[3]:>{w_crashed}}  "
        f"{headers[4]:>{w_empty}}  {headers[5]:>{w_size}}  {headers[6]}"
    )
    print()
    print(hdr)

    for dir_name, n_seeds, n_active, n_crashed, n_empty, size_str, status in rows:
        line = (
            f"{dir_name:<{w_dir}}  {n_seeds:>{w_seeds}}  "
            f"{n_active:>{w_active}}  {n_crashed:>{w_crashed}}  "
            f"{n_empty:>{w_empty}}  {size_str:>{w_size}}  {status}"
        )
        print(line)

    n_dirs = len(disk_entries)
    print(f"\n{n_dirs} directories ({format_bytes(total_bytes)})")


def _list_json(entries: list[ExperimentEntry]) -> None:
    """Structured JSON output."""
    entry_dicts = []
    total_seeds = 0
    total_bytes = 0
    labels_seen: set[str] = set()

    for e in entries:
        entry_dicts.append({
            "label": e.label,
            "exp_hash": e.exp_hash,
            "group": e.group,
            "source": e.source,
            "seeds_disk": len(e.seeds_disk),
            "seeds_consolidated": len(e.seeds_consolidated),
            "seeds_total": e.seed_count,
            "seeds_active": len(e.seeds_active),
            "seeds_crashed": len(e.seeds_crashed),
            "seeds_empty": len(e.seeds_empty),
            "disk_bytes": e.disk_bytes,
            "is_stale": e.is_stale,
        })
        total_seeds += e.seed_count
        total_bytes += e.disk_bytes
        labels_seen.add(e.label)

    output = {
        "entries": entry_dicts,
        "summary": {
            "labels": len(labels_seen),
            "configs": len(entries),
            "seeds": total_seeds,
            "disk_bytes": total_bytes,
        },
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# complete subcommand
# ---------------------------------------------------------------------------

def cmd_complete(args) -> None:
    """Find and execute missing experiment runs."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from run_all_experiments import (
        EXPERIMENT_GROUPS,
        check_experiment_exists,
        generate_all_runs,
        run_single_experiment,
    )

    groups = args.group if args.group else list(EXPERIMENT_GROUPS.keys())
    all_runs = generate_all_runs(groups, args.seeds)
    missing = [r for r in all_runs if not check_experiment_exists(r)]

    # Aggregate by label
    by_label: dict[str, dict[str, int]] = defaultdict(lambda: {"expected": 0, "missing": 0})
    for r in all_runs:
        by_label[r.label]["expected"] += 1
    for r in missing:
        by_label[r.label]["missing"] += 1

    # Print summary table
    total_expected = len(all_runs)
    total_missing = len(missing)
    total_existing = total_expected - total_missing

    if by_label:
        w_label = max(len("Label"), max(len(l) for l in by_label))
        print()
        print(f"{'Label':<{w_label}}  {'Expected':>8}  {'Existing':>8}  {'Missing':>7}")
        for label in sorted(by_label):
            info = by_label[label]
            existing = info["expected"] - info["missing"]
            print(
                f"{label:<{w_label}}  {info['expected']:>8}  "
                f"{existing:>8}  {info['missing']:>7}"
            )
        n_labels = len(by_label)
        print(f"\n{n_labels} labels, {total_expected} expected, "
              f"{total_existing} existing, {total_missing} missing")

    if not missing:
        print("\nAll runs complete!")
        return

    if args.dry_run:
        return

    # Execute missing runs
    print(f"\nExecuting {total_missing} missing runs "
          f"with {args.parallel} workers...")
    completed = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {
            executor.submit(run_single_experiment, r): r for r in missing
        }
        for future in as_completed(futures):
            run_id, success, message = future.result()
            if success:
                completed += 1
            else:
                failed += 1
                print(f"  FAIL: {run_id}: {message[:100]}")
            done = completed + failed
            print(f"[{done}/{total_missing}] ok={completed} fail={failed}")

    print(f"\nComplete: {completed} ok, {failed} failed")


# ---------------------------------------------------------------------------
# compact subcommand
# ---------------------------------------------------------------------------

def cmd_compact(args) -> None:
    """Compact experiment data into consolidated parquet files."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from consolidate import (
        _experiment_sort_key,
        scan_experiments,
    )

    if args.destructive and args.verify:
        print("Error: --destructive cannot be combined with --verify")
        sys.exit(1)

    # Safety check: refuse if active seeds
    store = ExperimentStore(base_dir=args.dir)
    store.scan()
    entries = store.get_entries(
        group=args.group if args.group else None,
        pattern=args.pattern,
        exclude=args.exclude,
    )
    active = [(e.label, e.exp_hash[:8], len(e.seeds_active))
              for e in entries if e.seeds_active]
    if active:
        print("Refusing to compact: active seeds detected:")
        for label, h, n in active:
            print(f"  {label}-{h}: {n} active")
        sys.exit(1)

    # Scan and filter experiments
    all_experiments = scan_experiments(args.dir)
    allowed = {(e.label, e.exp_hash) for e in entries}
    experiments = [e for e in all_experiments if (e[0], e[1]) in allowed]
    experiments.sort(key=lambda e: _experiment_sort_key(*e))

    if not experiments:
        print("No experiments to compact.")
        return

    total_seeds = sum(len(s) for _, _, _, s in experiments)
    print(f"Found {len(experiments)} experiments, {total_seeds} seeds")

    # Choose mode: partition when filtering subset, single-file otherwise
    is_subset = len(experiments) < len(all_experiments)
    if args.partition or is_subset:
        _compact_partitioned(experiments, args)
    else:
        _compact_incremental(experiments, args)

    # Verify if requested — scoped to compacted experiments only
    if args.verify:
        import random

        from consolidate import _verify_one_sample

        partitioned = args.partition or is_subset
        consolidated_path = str(Path(args.dir) / "consolidated.parquet")

        # Collect result files from compacted experiments only
        all_files = []
        for _, _, exp_dir, seed_dirs in experiments:
            for sd in seed_dirs:
                rp = sd / "results.parquet"
                if rp.exists():
                    all_files.append(rp)

        sample = random.sample(all_files, min(args.verify_sample, len(all_files)))
        print(f"\nVerifying {len(sample)} / {len(all_files)} files...")
        passed = failed = 0
        for fp in sample:
            try:
                result = _verify_one_sample(consolidated_path, args.dir, fp, partitioned)
                if result is None:
                    continue
                if result[0] == "pass":
                    passed += 1
                else:
                    failed += 1
                    if result[1]:
                        print(f"  {result[1]}")
            except Exception as exc:
                print(f"  ERROR: {fp.parent.parent.name}/{fp.parent.name}: {exc}")
                failed += 1
        print(f"Verification: {passed} passed, {failed} failed")


def _compact_incremental(experiments, args) -> None:
    """Single-file consolidation."""
    from consolidate import (
        _consolidate_experiments,
        discover_schema,
    )

    output_path = str(Path(args.dir) / "consolidated.parquet")
    print(f"Writing {output_path} ({args.compression}, level {args.compression_level})")

    schema = discover_schema(experiments)
    total_rows, deleted_dirs, deleted_bytes = _consolidate_experiments(
        experiments, output_path, schema,
        args.compression, args.compression_level, args.destructive,
    )

    file_size = Path(output_path).stat().st_size
    print(f"Wrote {total_rows:,} rows ({format_bytes(file_size)})")
    if args.destructive and deleted_dirs:
        print(f"Deleted {deleted_dirs} directories ({format_bytes(deleted_bytes)} freed)")


def _compact_partitioned(experiments, args) -> None:
    """Per-label partitioned consolidation."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from consolidate import _consolidate_group

    # Group by label
    groups: dict[str, list] = defaultdict(list)
    for exp_name, exp_hash, exp_dir, seed_dirs in experiments:
        groups[exp_name].append((exp_name, exp_hash, exp_dir, seed_dirs))

    print(f"Compacting {len(groups)} labels in partitioned mode")

    # Build work items — convert Paths to strings for pickling
    work_items = []
    for group_name in sorted(groups):
        output_path = str(Path(args.dir) / f"{group_name}.parquet")
        tuples = [
            (n, h, str(d), [str(s) for s in seeds])
            for n, h, d, seeds in groups[group_name]
        ]
        work_items.append((
            group_name, tuples, output_path,
            args.compression, args.compression_level, args.destructive,
        ))

    total_rows = 0
    total_deleted = 0
    total_freed = 0
    failures = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(_consolidate_group, item): item[0]
            for item in work_items
        }
        for future in as_completed(futures):
            name, rows, deleted, freed, out_path, error = future.result()
            if error:
                print(f"  FAIL {name}: {error}")
                failures.append(name)
            else:
                file_size = Path(out_path).stat().st_size
                print(f"  {name}: {rows:,} rows ({format_bytes(file_size)})")
                total_rows += rows
                total_deleted += deleted
                total_freed += freed

    print(f"\nWrote {total_rows:,} rows across {len(groups) - len(failures)} labels")
    if failures:
        print(f"Failed: {', '.join(failures)}")
    if args.destructive and total_deleted:
        print(f"Deleted {total_deleted} directories ({format_bytes(total_freed)} freed)")


# ---------------------------------------------------------------------------
# gc subcommand
# ---------------------------------------------------------------------------

def cmd_gc(args) -> None:
    """Garbage collect stale and incomplete experiment data."""
    import shutil

    store = ExperimentStore(base_dir=args.dir)
    store.scan()

    entries = store.get_entries(
        group=args.group if args.group else None,
        pattern=args.pattern,
        exclude=args.exclude,
    )

    # Default: both modes if neither flag specified
    do_stale = args.stale or not args.incomplete
    do_incomplete = args.incomplete or not args.stale

    targets: list[tuple[str, Path, int]] = []  # (reason, path, bytes)

    for e in entries:
        if e.disk_dir is None:
            continue

        # Stale: entire directory has hash mismatch
        if do_stale and e.is_stale:
            targets.append(("stale", e.disk_dir, e.disk_bytes))
            continue  # whole dir; skip per-seed checks

        # Incomplete: empty or crashed seed dirs (never active)
        if do_incomplete:
            for seed in sorted(e.seeds_crashed):
                seed_dir = e.disk_dir / str(seed)
                if seed_dir.is_dir():
                    try:
                        sz = sum(f.stat().st_size for f in seed_dir.rglob("*") if f.is_file())
                    except OSError:
                        sz = 0
                    targets.append(("crashed", seed_dir, sz))
            for seed in sorted(e.seeds_empty):
                seed_dir = e.disk_dir / str(seed)
                if seed_dir.is_dir():
                    targets.append(("empty", seed_dir, 0))

    if not targets:
        print("Nothing to clean up.")
        return

    # Print what would be deleted
    total_bytes = 0
    w_reason = max(len("Reason"), max(len(t[0]) for t in targets))
    w_path = max(len("Path"), max(len(str(t[1])) for t in targets))

    print()
    print(f"{'Reason':<{w_reason}}  {'Path':<{w_path}}  {'Size':>10}")
    for reason, path, sz in targets:
        print(f"{reason:<{w_reason}}  {str(path):<{w_path}}  {format_bytes(sz):>10}")
        total_bytes += sz

    print(f"\n{len(targets)} targets ({format_bytes(total_bytes)})")

    if not args.force:
        print("\nDry run. Use --force to delete.")
        return

    # Execute deletions
    deleted = 0
    freed = 0
    for reason, path, sz in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            deleted += 1
            freed += sz
        except OSError as exc:
            print(f"  ERROR: {path}: {exc}")

    print(f"\nDeleted {deleted}/{len(targets)} ({format_bytes(freed)} freed)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="expctl",
        description="Experiment management utility",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    # list subcommand
    sub_list = subparsers.add_parser("list", help="List experiments")
    sub_list.add_argument("--group", action="append", help="Filter by group (repeatable)")
    sub_list.add_argument("--pattern", help="fnmatch pattern on label")
    sub_list.add_argument("--exclude", help="fnmatch pattern to exclude")
    mode = sub_list.add_mutually_exclusive_group()
    mode.add_argument("--verbose", "-v", action="store_true", help="Per-hash detail")
    mode.add_argument("--physical", action="store_true", help="Raw directory layout")
    mode.add_argument("--json", action="store_true", help="JSON output")
    sub_list.add_argument("--stale", action="store_true", help="Only stale experiments")
    sub_list.add_argument("--dir", default="experiments", help="Base directory")
    sub_list.set_defaults(func=cmd_list)

    sub_compact = subparsers.add_parser("compact", help="Compact experiments to parquet")
    sub_compact.add_argument("--group", action="append", help="Filter by group (repeatable)")
    sub_compact.add_argument("--pattern", help="fnmatch pattern on label")
    sub_compact.add_argument("--exclude", help="fnmatch pattern to exclude")
    sub_compact.add_argument("--partition", action="store_true",
                             help="Per-label parquet files (auto when filtering)")
    sub_compact.add_argument("--destructive", action="store_true",
                             help="Delete source dirs after compaction")
    sub_compact.add_argument("--verify", action="store_true",
                             help="Verify after compaction")
    sub_compact.add_argument("--verify-sample", type=int, default=20)
    sub_compact.add_argument("--compression", default="zstd", choices=["zstd", "snappy"])
    sub_compact.add_argument("--compression-level", type=int, default=3)
    sub_compact.add_argument("--max-workers", type=int, default=None)
    sub_compact.add_argument("--dir", default="experiments", help="Base directory")
    sub_compact.set_defaults(func=cmd_compact)

    sub_gc = subparsers.add_parser("gc", help="Garbage collect experiments")
    sub_gc.add_argument("--group", action="append", help="Filter by group (repeatable)")
    sub_gc.add_argument("--pattern", help="fnmatch pattern on label")
    sub_gc.add_argument("--exclude", help="fnmatch pattern to exclude")
    sub_gc.add_argument("--stale", action="store_true", help="Remove stale experiment dirs")
    sub_gc.add_argument("--incomplete", action="store_true", help="Remove empty/crashed seed dirs")
    sub_gc.add_argument("--force", action="store_true", help="Actually delete (default is dry run)")
    sub_gc.add_argument("--dir", default="experiments", help="Base directory")
    sub_gc.set_defaults(func=cmd_gc)

    sub_complete = subparsers.add_parser("complete", help="Complete missing seeds")
    sub_complete.add_argument("--group", action="append", help="Filter by group (repeatable)")
    sub_complete.add_argument("--seeds", type=int, required=True, help="Seeds per configuration")
    sub_complete.add_argument("--parallel", type=int, default=4, help="Parallel workers")
    sub_complete.add_argument("--dry-run", action="store_true", help="Report missing only")
    sub_complete.add_argument("--dir", default="experiments", help="Base directory")
    sub_complete.set_defaults(func=cmd_complete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
