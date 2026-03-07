"""Tests for scripts/expctl.py — ExperimentStore core."""
from __future__ import annotations

import json
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Ensure scripts/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from endive.config import compute_experiment_hash
from scripts.expctl import ExperimentEntry, ExperimentStore, format_bytes, _build_label_to_group


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_GROUPS = {
    "baseline": ["exp1_fa_baseline.toml"],
    "heatmap": ["exp2_mix_heatmap.toml"],
    "tables": ["exp4a_tables_fa.toml", "exp4b_tables_mix.toml"],
}


def _write_cfg_toml(exp_dir: Path, config: dict) -> None:
    """Write a cfg.toml file."""
    import tomli_w
    with open(exp_dir / "cfg.toml", "wb") as f:
        tomli_w.dump(config, f)


def _write_cfg_toml_manual(exp_dir: Path, config: dict) -> None:
    """Write a simple cfg.toml without tomli_w dependency."""
    lines = []
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, dict):
                    lines.append(f"[{section}.{k}]")
                    for k2, v2 in v.items():
                        lines.append(f"{k2} = {_toml_value(v2)}")
                else:
                    lines.append(f"{k} = {_toml_value(v)}")
        else:
            lines.append(f"{section} = {_toml_value(values)}")
    (exp_dir / "cfg.toml").write_text("\n".join(lines) + "\n")


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, int):
        return str(v)
    elif isinstance(v, float):
        return str(v)
    elif isinstance(v, str):
        return f'"{v}"'
    else:
        return str(v)


def _write_version_txt(exp_dir: Path, code_hash: str = "abcdef1234567890") -> None:
    (exp_dir / "version.txt").write_text(f"git_sha=unknown\ncode_hash={code_hash}\n")


def _make_experiment(
    base_dir: Path,
    label: str,
    exp_hash: str,
    seeds: list[int] | None = None,
    empty_seeds: list[int] | None = None,
    active_seeds: list[int] | None = None,
    crashed_seeds: list[int] | None = None,
    config: dict | None = None,
    code_hash: str = "abcdef1234567890",
) -> Path:
    """Create a fake experiment directory structure."""
    exp_dir = base_dir / f"{label}-{exp_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if config:
        _write_cfg_toml_manual(exp_dir, config)

    _write_version_txt(exp_dir, code_hash)

    for seed in (seeds or []):
        seed_dir = exp_dir / str(seed)
        seed_dir.mkdir(exist_ok=True)
        # Write a minimal parquet file
        table = pa.table({"x": [1, 2, 3]})
        pq.write_table(table, seed_dir / "results.parquet")

    for seed in (empty_seeds or []):
        seed_dir = exp_dir / str(seed)
        seed_dir.mkdir(exist_ok=True)
        # No files — simulates interrupted run

    for seed in (active_seeds or []):
        seed_dir = exp_dir / str(seed)
        seed_dir.mkdir(exist_ok=True)
        # .running.parquet exists, .progress.json is recent
        table = pa.table({"x": [1]})
        pq.write_table(table, seed_dir / ".running.parquet")
        progress = {"sim_time_ms": 1000, "committed": 10}
        (seed_dir / ".progress.json").write_text(json.dumps(progress))

    for seed in (crashed_seeds or []):
        seed_dir = exp_dir / str(seed)
        seed_dir.mkdir(exist_ok=True)
        # .running.parquet exists, .progress.json is old or missing
        table = pa.table({"x": [1]})
        pq.write_table(table, seed_dir / ".running.parquet")
        # Set old mtime to simulate stale progress
        progress_path = seed_dir / ".progress.json"
        progress_path.write_text(json.dumps({"sim_time_ms": 500}))
        # Set mtime to 2 hours ago
        old_time = time.time() - 7200
        import os
        os.utime(progress_path, (old_time, old_time))

    return exp_dir


def _make_consolidated(base_dir: Path, entries: list[tuple[str, str, list[int]]]) -> Path:
    """Create a fake consolidated.parquet with given (label, hash, seeds) entries."""
    exp_names = []
    exp_hashes = []
    seeds = []
    # Need at least one more column to make it a valid table
    xs = []

    for label, exp_hash, seed_list in entries:
        for s in seed_list:
            exp_names.append(label)
            exp_hashes.append(exp_hash)
            seeds.append(s)
            xs.append(0)

    table = pa.table({
        "exp_name": exp_names,
        "exp_hash": exp_hashes,
        "seed": seeds,
        "x": xs,
    })
    path = base_dir / "consolidated.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def mock_groups():
    """Patch _load_experiment_groups to return test groups."""
    with patch("scripts.expctl._load_experiment_groups", return_value=_MOCK_GROUPS):
        yield


# ---------------------------------------------------------------------------
# Unit tests: ExperimentEntry
# ---------------------------------------------------------------------------

class TestExperimentEntry:
    def test_all_seeds_union(self):
        entry = ExperimentEntry(
            label="test", exp_hash="aabbccdd", group="g",
            source="both", disk_dir=None, config=None,
            seeds_disk={1, 2, 3}, seeds_consolidated={2, 3, 4},
        )
        assert entry.all_seeds == {1, 2, 3, 4}
        assert entry.seed_count == 4

    def test_empty_entry(self):
        entry = ExperimentEntry(
            label="test", exp_hash="aabbccdd", group="g",
            source="disk", disk_dir=None, config=None,
        )
        assert entry.all_seeds == set()
        assert entry.seed_count == 0


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_format_bytes(self):
        assert format_bytes(500) == "500 B"
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(int(1.5 * 1024 * 1024)) == "1.5 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(int(2.5 * 1024 * 1024 * 1024)) == "2.5 GB"

    def test_build_label_to_group(self):
        mapping = _build_label_to_group(_MOCK_GROUPS)
        assert mapping["exp1_fa_baseline"] == "baseline"
        assert mapping["exp2_mix_heatmap"] == "heatmap"
        assert mapping["exp4a_tables_fa"] == "tables"
        assert mapping["exp4b_tables_mix"] == "tables"

    def test_unknown_label_group(self, tmp_path, mock_groups):
        store = ExperimentStore(base_dir=tmp_path)
        assert store._label_group("exp_unknown") == "unknown"
        assert store._label_group("exp1_fa_baseline") == "baseline"


# ---------------------------------------------------------------------------
# Unit tests: _scan_disk
# ---------------------------------------------------------------------------

class TestScanDisk:
    def test_complete_seeds(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd", seeds=[42, 43, 44])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entries = store.get_entries()
        assert len(entries) == 1

        entry = entries[0]
        assert entry.label == "exp1_fa_baseline"
        assert entry.exp_hash == "aabbccdd"
        assert entry.group == "baseline"
        assert entry.source == "disk"
        assert entry.seeds_disk == {42, 43, 44}
        assert entry.seed_count == 3
        assert entry.disk_bytes > 0

    def test_empty_seeds(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd",
                         seeds=[42], empty_seeds=[99, 100])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entry = store.get_entries()[0]
        assert entry.seeds_disk == {42}
        assert entry.seeds_empty == {99, 100}

    def test_active_seeds(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd",
                         seeds=[42], active_seeds=[50])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entry = store.get_entries()[0]
        assert entry.seeds_disk == {42}
        assert entry.seeds_active == {50}
        assert entry.seeds_crashed == set()

    def test_crashed_seeds(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd",
                         seeds=[42], crashed_seeds=[60])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entry = store.get_entries()[0]
        assert entry.seeds_disk == {42}
        assert entry.seeds_crashed == {60}
        assert entry.seeds_active == set()

    def test_crashed_no_progress(self, tmp_path, mock_groups):
        """Seed with .running.parquet but no .progress.json = crashed."""
        exp_dir = tmp_path / "exp1_fa_baseline-aabbccdd"
        exp_dir.mkdir()
        _write_version_txt(exp_dir)

        seed_dir = exp_dir / "70"
        seed_dir.mkdir()
        table = pa.table({"x": [1]})
        pq.write_table(table, seed_dir / ".running.parquet")
        # No .progress.json

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entry = store.get_entries()[0]
        assert entry.seeds_crashed == {70}
        assert entry.seeds_active == set()

    def test_code_version_parsed(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd",
                         seeds=[42], code_hash="deadbeef12345678")

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entry = store.get_entries()[0]
        assert entry.code_version == "deadbeef12345678"

    def test_skips_non_experiment_dirs(self, tmp_path, mock_groups):
        # No hyphen
        (tmp_path / "not_an_experiment").mkdir()
        # Short hash
        (tmp_path / "foo-abc").mkdir()
        # Non-hex hash
        (tmp_path / "foo-gggggg").mkdir()
        # Valid
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd", seeds=[42])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        assert len(store.get_entries()) == 1

    def test_missing_base_dir(self, tmp_path, mock_groups):
        store = ExperimentStore(base_dir=tmp_path / "nonexistent")
        store._scan_disk()
        assert store.get_entries() == []

    def test_multiple_experiments(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd", seeds=[42, 43])
        _make_experiment(tmp_path, "exp1_fa_baseline", "11223344", seeds=[42])
        _make_experiment(tmp_path, "exp2_mix_heatmap", "55667788", seeds=[42, 43, 44])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()

        entries = store.get_entries()
        assert len(entries) == 3

        labels = [e.label for e in entries]
        assert labels.count("exp1_fa_baseline") == 2
        assert labels.count("exp2_mix_heatmap") == 1


# ---------------------------------------------------------------------------
# Unit tests: _scan_consolidated
# ---------------------------------------------------------------------------

class TestScanConsolidated:
    def test_consolidated_only(self, tmp_path, mock_groups):
        _make_consolidated(tmp_path, [
            ("exp1_fa_baseline", "aabbccdd", [42, 43, 44]),
            ("exp2_mix_heatmap", "55667788", [42, 43]),
        ])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_consolidated()

        entries = store.get_entries()
        assert len(entries) == 2

        e1 = [e for e in entries if e.label == "exp1_fa_baseline"][0]
        assert e1.seeds_consolidated == {42, 43, 44}
        assert e1.source == "consolidated"
        assert e1.disk_dir is None

    def test_merge_disk_and_consolidated(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd", seeds=[42, 43])
        _make_consolidated(tmp_path, [
            ("exp1_fa_baseline", "aabbccdd", [42, 43, 44, 45]),
        ])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()
        store._scan_consolidated()

        entries = store.get_entries()
        assert len(entries) == 1

        entry = entries[0]
        assert entry.source == "both"
        assert entry.seeds_disk == {42, 43}
        assert entry.seeds_consolidated == {42, 43, 44, 45}
        assert entry.all_seeds == {42, 43, 44, 45}
        assert entry.seed_count == 4

    def test_per_label_parquet(self, tmp_path, mock_groups):
        """Per-label parquet files (from partitioned compaction) are also scanned."""
        # Create a per-label parquet (not named consolidated.parquet)
        table = pa.table({
            "exp_name": ["exp1_fa_baseline", "exp1_fa_baseline"],
            "exp_hash": ["aabbccdd", "aabbccdd"],
            "seed": [42, 43],
            "x": [0, 0],
        })
        pq.write_table(table, tmp_path / "exp1_fa_baseline.parquet")

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_consolidated()

        entries = store.get_entries()
        assert len(entries) == 1
        assert entries[0].seeds_consolidated == {42, 43}


# ---------------------------------------------------------------------------
# Unit tests: _detect_stale_hashes
# ---------------------------------------------------------------------------

class TestStaleDetection:
    def test_matching_hash_not_stale(self, tmp_path, mock_groups):
        """Entry with config that produces same hash = not stale."""
        config = {
            "simulation": {"duration_ms": 3600000, "seed": 42},
            "experiment": {"label": "exp1_fa_baseline"},
            "catalog": {"num_tables": 1},
            "transaction": {"retry": 10},
        }
        # Compute the expected hash
        expected_hash = compute_experiment_hash(config)

        _make_experiment(tmp_path, "exp1_fa_baseline", expected_hash,
                         seeds=[42], config=config)

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()
        store._detect_stale_hashes()

        entry = store.get_entries()[0]
        assert not entry.is_stale

    def test_mismatched_hash_is_stale(self, tmp_path, mock_groups):
        """Entry with config that produces different hash = stale."""
        config = {
            "simulation": {"duration_ms": 3600000, "seed": 42},
            "experiment": {"label": "exp1_fa_baseline"},
            "catalog": {"num_tables": 1},
            "transaction": {"retry": 10},
        }

        _make_experiment(tmp_path, "exp1_fa_baseline", "deadbeef",
                         seeds=[42], config=config)

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()
        store._detect_stale_hashes()

        entry = store.get_entries()[0]
        assert entry.is_stale

    def test_no_config_with_old_code_hash(self, tmp_path, mock_groups):
        """No cfg.toml but code_version differs from current = stale."""
        exp_dir = tmp_path / "exp1_fa_baseline-aabbccdd"
        exp_dir.mkdir()
        _write_version_txt(exp_dir, "0000000000000000")
        seed_dir = exp_dir / "42"
        seed_dir.mkdir()
        table = pa.table({"x": [1]})
        pq.write_table(table, seed_dir / "results.parquet")

        with patch("scripts.expctl.compute_code_hash", return_value="1111111111111111"):
            store = ExperimentStore(base_dir=tmp_path)
            store._scan_disk()
            store._detect_stale_hashes()

        entry = store.get_entries()[0]
        assert entry.is_stale

    def test_no_config_matching_code_hash(self, tmp_path, mock_groups):
        """No cfg.toml but code_version matches current = not stale."""
        exp_dir = tmp_path / "exp1_fa_baseline-aabbccdd"
        exp_dir.mkdir()
        _write_version_txt(exp_dir, "matchingcodehash")
        seed_dir = exp_dir / "42"
        seed_dir.mkdir()
        table = pa.table({"x": [1]})
        pq.write_table(table, seed_dir / "results.parquet")

        with patch("scripts.expctl.compute_code_hash", return_value="matchingcodehash"):
            store = ExperimentStore(base_dir=tmp_path)
            store._scan_disk()
            store._detect_stale_hashes()

        entry = store.get_entries()[0]
        assert not entry.is_stale


# ---------------------------------------------------------------------------
# Unit tests: filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    @pytest.fixture
    def store_with_entries(self, tmp_path, mock_groups):
        _make_experiment(tmp_path, "exp1_fa_baseline", "aabbccdd", seeds=[42])
        _make_experiment(tmp_path, "exp1_fa_baseline", "11223344", seeds=[42])
        _make_experiment(tmp_path, "exp2_mix_heatmap", "55667788", seeds=[42])
        _make_experiment(tmp_path, "exp4a_tables_fa", "99aabbcc", seeds=[42])

        store = ExperimentStore(base_dir=tmp_path)
        store._scan_disk()
        return store

    def test_filter_by_group(self, store_with_entries):
        entries = store_with_entries.get_entries(group="baseline")
        assert len(entries) == 2
        assert all(e.group == "baseline" for e in entries)

    def test_filter_by_multiple_groups(self, store_with_entries):
        entries = store_with_entries.get_entries(group=["baseline", "heatmap"])
        assert len(entries) == 3

    def test_filter_by_pattern(self, store_with_entries):
        entries = store_with_entries.get_entries(pattern="exp1_*")
        assert len(entries) == 2
        assert all(e.label == "exp1_fa_baseline" for e in entries)

    def test_filter_by_exclude(self, store_with_entries):
        entries = store_with_entries.get_entries(exclude="exp1_*")
        assert len(entries) == 2
        labels = {e.label for e in entries}
        assert "exp1_fa_baseline" not in labels

    def test_filter_combined(self, store_with_entries):
        entries = store_with_entries.get_entries(group="baseline", pattern="exp1_*")
        assert len(entries) == 2

    def test_no_match(self, store_with_entries):
        entries = store_with_entries.get_entries(pattern="nonexistent_*")
        assert entries == []

    def test_sorted_output(self, store_with_entries):
        entries = store_with_entries.get_entries()
        groups = [e.group for e in entries]
        labels = [e.label for e in entries]
        # Should be sorted by (group, label, hash)
        assert groups == sorted(groups)

    def test_labels_grouping(self, store_with_entries):
        by_label = store_with_entries.labels()
        assert "exp1_fa_baseline" in by_label
        assert len(by_label["exp1_fa_baseline"]) == 2
        assert "exp2_mix_heatmap" in by_label
        assert len(by_label["exp2_mix_heatmap"]) == 1


# ---------------------------------------------------------------------------
# Integration: full scan
# ---------------------------------------------------------------------------

class TestFullScan:
    def test_scan_merges_all_sources(self, tmp_path, mock_groups):
        """Full scan() integrates disk + consolidated + stale detection."""
        config = {
            "simulation": {"duration_ms": 3600000, "seed": 42},
            "experiment": {"label": "exp1_fa_baseline"},
            "catalog": {"num_tables": 1},
            "transaction": {"retry": 10},
        }
        expected_hash = compute_experiment_hash(config)

        _make_experiment(tmp_path, "exp1_fa_baseline", expected_hash,
                         seeds=[42, 43], config=config)
        _make_consolidated(tmp_path, [
            ("exp1_fa_baseline", expected_hash, [42, 43, 44]),
        ])

        store = ExperimentStore(base_dir=tmp_path)
        store.scan()

        entries = store.get_entries()
        assert len(entries) == 1

        entry = entries[0]
        assert entry.source == "both"
        assert entry.seeds_disk == {42, 43}
        assert entry.seeds_consolidated == {42, 43, 44}
        assert entry.all_seeds == {42, 43, 44}
        assert not entry.is_stale

    def test_scan_empty_dir(self, tmp_path, mock_groups):
        store = ExperimentStore(base_dir=tmp_path)
        store.scan()
        assert store.get_entries() == []

    def test_scan_with_mixed_seed_states(self, tmp_path, mock_groups):
        """Experiment with all seed states: complete, empty, active, crashed."""
        _make_experiment(
            tmp_path, "exp1_fa_baseline", "aabbccdd",
            seeds=[42, 43],
            empty_seeds=[99],
            active_seeds=[50],
            crashed_seeds=[60],
        )

        store = ExperimentStore(base_dir=tmp_path)
        store.scan()

        entry = store.get_entries()[0]
        assert entry.seeds_disk == {42, 43}
        assert entry.seeds_empty == {99}
        assert entry.seeds_active == {50}
        assert entry.seeds_crashed == {60}
        assert entry.all_seeds == {42, 43}  # active/crashed/empty not in all_seeds
