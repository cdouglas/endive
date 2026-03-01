"""Tests for scan_all_experiments() deduplication and multi-hash handling.

Verifies that:
1. Disk experiments take precedence over consolidated for same (label, hash)
2. Different hashes for the same label remain separate entries
3. Consolidated-only experiments appear when no disk directory exists
4. Row-group metadata scanning works correctly for discovery
5. Downstream pipelines (index, heatmap, op-type) keep hashes separate
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import endive.saturation_analysis as sa
from endive.saturation_analysis import (
    scan_all_experiments,
    scan_consolidated_experiments,
    scan_experiment_directories,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CFG = """\
[simulation]
duration_ms = 3600000
output_path = "results.parquet"

[experiment]
label = "{label}"

[catalog]
num_tables = {num_tables}
mode = "instant"

[storage]
provider = "instant"

[transaction]
retry = 10
inter_arrival.scale = {scale}
real_conflict_probability = {p_real}
runtime.mean = 10000
runtime.sigma = 1.5
operation_types.fast_append = {fa_ratio}
operation_types.validated_overwrite = {vo_ratio}
"""


def _make_cfg(label="exp_test", num_tables=1, scale=100.0, p_real=0.0,
              fa_ratio=1.0, vo_ratio=0.0):
    return _BASE_CFG.format(
        label=label, num_tables=num_tables, scale=scale, p_real=p_real,
        fa_ratio=fa_ratio, vo_ratio=vo_ratio,
    )


def _make_results(n=100, seed_val=42):
    """Create a minimal transaction results DataFrame."""
    rng = np.random.RandomState(seed_val)
    return pd.DataFrame({
        "txn_id": range(n),
        "t_submit": np.linspace(0, 3_600_000, n).astype(int),
        "t_commit": (np.linspace(0, 3_600_000, n) + 1000).astype(int),
        "commit_latency": rng.randint(500, 5000, n),
        "total_latency": rng.randint(5000, 15000, n),
        "n_retries": rng.randint(0, 3, n).astype(np.int8),
        "status": ["committed"] * n,
        "operation_type": rng.choice(["fast_append", "validated_overwrite"], n),
    })


def _create_disk_experiment(base_dir, label, exp_hash, seeds=(42, 43, 44),
                            n_txns=100, **cfg_kwargs):
    """Create an experiment directory on disk with cfg.toml and seed dirs."""
    exp_dir = Path(base_dir) / f"{label}-{exp_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "cfg.toml", "w") as f:
        f.write(_make_cfg(label=label, **cfg_kwargs))
    for s in seeds:
        seed_dir = exp_dir / str(s)
        seed_dir.mkdir(exist_ok=True)
        _make_results(n_txns, seed_val=s).to_parquet(seed_dir / "results.parquet")
    return exp_dir


def _flatten_config(config_text):
    """Flatten TOML text into (key, value) pairs for the consolidated config column."""
    import tomli
    cfg = tomli.loads(config_text)
    items = []

    def _flatten(d, prefix=""):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _flatten(v, full_key)
            else:
                items.append((full_key, str(v)))

    _flatten(cfg)
    return items


def _create_consolidated(base_dir, experiments, filename="consolidated.parquet"):
    """Write a consolidated parquet with proper row-group-per-seed structure.

    ``experiments`` is a list of dicts:
        {label, hash, seeds: [int], cfg_text: str, n_txns: int}
    """
    path = os.path.join(base_dir, filename)
    writer = None
    schema = None

    for exp in experiments:
        config_map = _flatten_config(exp["cfg_text"])
        for seed_val in exp["seeds"]:
            df = _make_results(exp.get("n_txns", 100), seed_val=seed_val)
            df["exp_name"] = exp["label"]
            df["exp_hash"] = exp["hash"]
            df["seed"] = seed_val
            df["config"] = [config_map] * len(df)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    path, schema,
                    use_dictionary=["exp_name", "exp_hash", "status"],
                    write_statistics=True,
                )
            else:
                table = table.cast(schema)
            writer.write_table(table)

    if writer is not None:
        writer.close()
    return path


# ---------------------------------------------------------------------------
# scan_all_experiments dedup
# ---------------------------------------------------------------------------


class TestScanAllExperimentsDedup:
    """Verify disk-vs-consolidated deduplication in scan_all_experiments()."""

    def test_disk_takes_precedence_over_consolidated(self):
        """Same (label, hash) on disk and consolidated: disk wins, only one entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label, hsh = "exp_test", "aabbcc11"
            cfg_text = _make_cfg(label=label, scale=100.0)

            _create_disk_experiment(tmpdir, label, hsh, seeds=(42, 43, 44))
            _create_consolidated(tmpdir, [
                {"label": label, "hash": hsh, "seeds": [42, 43, 44],
                 "cfg_text": cfg_text},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_all_experiments(tmpdir, f"{label}-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 1, f"Expected 1 experiment, got {len(exps)}: {list(exps.keys())}"
            key = f"{label}-{hsh}"
            assert key in exps
            # disk entry has a non-None dir
            assert exps[key]["dir"] is not None

    def test_consolidated_only_when_no_disk(self):
        """Experiment in consolidated but not on disk: appears with dir=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label, hsh = "exp_test", "aabbcc11"
            cfg_text = _make_cfg(label=label, scale=100.0)

            _create_consolidated(tmpdir, [
                {"label": label, "hash": hsh, "seeds": [42, 43, 44],
                 "cfg_text": cfg_text},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_all_experiments(tmpdir, f"{label}-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 1
            key = f"{label}-{hsh}"
            assert key in exps
            assert exps[key]["dir"] is None
            assert exps[key]["label"] == label
            assert exps[key]["hash"] == hsh
            assert sorted(exps[key]["seeds"]) == [42, 43, 44]

    def test_no_duplicate_when_both_sources_present(self):
        """Regression test for P2 key-mismatch bug.

        Before the fix, scan_experiment_directories() keyed by full path
        and scan_consolidated_experiments() keyed by short name, so the
        dedup check always failed and every experiment appeared twice.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            label, hsh = "exp_dup_check", "deadbeef"
            cfg_text = _make_cfg(label=label, scale=50.0)

            _create_disk_experiment(tmpdir, label, hsh, seeds=(42, 43, 44))
            _create_consolidated(tmpdir, [
                {"label": label, "hash": hsh, "seeds": [42, 43, 44],
                 "cfg_text": cfg_text},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_all_experiments(tmpdir, f"{label}-*")
            finally:
                sa.CONFIG = old_config

            # Must be exactly 1, not 2
            assert len(exps) == 1, (
                f"Dedup failed: got {len(exps)} entries for same (label, hash). "
                f"Keys: {list(exps.keys())}"
            )

    def test_multiple_experiments_both_sources_dedup(self):
        """Multiple experiments, some on disk+consolidated, some consolidated-only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_a = _make_cfg(label="exp_multi", scale=100.0)
            cfg_b = _make_cfg(label="exp_multi", scale=200.0)
            cfg_c = _make_cfg(label="exp_multi", scale=300.0)

            # A on disk + consolidated, B on disk only, C consolidated only
            _create_disk_experiment(tmpdir, "exp_multi", "hash_aaa", seeds=(42, 43, 44), scale=100.0)
            _create_disk_experiment(tmpdir, "exp_multi", "hash_bbb", seeds=(42, 43, 44), scale=200.0)
            _create_consolidated(tmpdir, [
                {"label": "exp_multi", "hash": "hash_aaa", "seeds": [42, 43, 44],
                 "cfg_text": cfg_a},
                {"label": "exp_multi", "hash": "hash_ccc", "seeds": [42, 43, 44],
                 "cfg_text": cfg_c},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_all_experiments(tmpdir, "exp_multi-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 3, f"Expected 3 experiments, got {len(exps)}: {list(exps.keys())}"
            assert exps["exp_multi-hash_aaa"]["dir"] is not None  # disk wins
            assert exps["exp_multi-hash_bbb"]["dir"] is not None  # disk only
            assert exps["exp_multi-hash_ccc"]["dir"] is None       # consolidated only


# ---------------------------------------------------------------------------
# Multi-hash separation (different configs produce different hashes)
# ---------------------------------------------------------------------------


class TestMultiHashSeparation:
    """Different hashes for the same label must remain separate throughout."""

    def _setup_multi_hash(self, tmpdir):
        """Create two experiments with same label, different hashes (different params)."""
        _create_disk_experiment(
            tmpdir, "exp_fa_baseline", "hash_s100",
            seeds=(42, 43, 44), scale=100.0,
        )
        _create_disk_experiment(
            tmpdir, "exp_fa_baseline", "hash_s200",
            seeds=(42, 43, 44), scale=200.0,
        )
        return tmpdir

    def test_scan_keeps_hashes_separate(self):
        """scan_all_experiments returns separate entries per hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_multi_hash(tmpdir)

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_all_experiments(tmpdir, "exp_fa_baseline-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 2
            assert "exp_fa_baseline-hash_s100" in exps
            assert "exp_fa_baseline-hash_s200" in exps

    def test_build_index_separate_rows_per_hash(self):
        """build_experiment_index creates one row per (label, hash), not per label."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_multi_hash(tmpdir)

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            sa.CONFIG["analysis"]["min_seeds"] = 1
            try:
                index_df = sa.build_experiment_index(tmpdir, "exp_fa_baseline-*")
            finally:
                sa.CONFIG = old_config

            assert len(index_df) == 2, f"Expected 2 rows, got {len(index_df)}"
            # Both rows have the same label but different hashes
            assert set(index_df["label"]) == {"exp_fa_baseline"}
            assert len(index_df["hash"].unique()) == 2
            # Different inter_arrival_scale values
            scales = sorted(index_df["inter_arrival_scale"].tolist())
            assert scales == [100.0, 200.0]

    def test_heatmap_params_separate_per_hash(self):
        """_extract_heatmap_params returns separate records per hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_multi_hash(tmpdir)

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            sa.CONFIG["analysis"]["min_seeds"] = 1
            try:
                params_df = sa._extract_heatmap_params(
                    tmpdir, "exp_fa_baseline-*",
                    x_param="inter_arrival_scale",
                    y_param="num_tables",
                )
            finally:
                sa.CONFIG = old_config

            assert len(params_df) == 2, f"Expected 2 records, got {len(params_df)}"
            scales = sorted(params_df["inter_arrival_scale"].tolist())
            assert scales == [100.0, 200.0]

    def test_op_type_analysis_separate_per_hash(self):
        """_analyze_op_type_experiments keeps hashes separate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_disk_experiment(
                tmpdir, "exp_mix", "hash_fa50",
                seeds=(42,), scale=100.0, fa_ratio=0.5, vo_ratio=0.5,
            )
            _create_disk_experiment(
                tmpdir, "exp_mix", "hash_fa90",
                seeds=(42,), scale=100.0, fa_ratio=0.9, vo_ratio=0.1,
            )

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            sa.CONFIG["analysis"]["min_seeds"] = 1
            try:
                df = sa._analyze_op_type_experiments(tmpdir, "exp_mix-*")
            finally:
                sa.CONFIG = old_config

            # Should have records for both fa_ratios
            assert len(df) > 0
            fa_ratios = sorted(df["fa_ratio"].unique())
            assert fa_ratios == [0.5, 0.9], f"Expected [0.5, 0.9], got {fa_ratios}"


# ---------------------------------------------------------------------------
# Consolidated multi-hash separation
# ---------------------------------------------------------------------------


class TestConsolidatedMultiHash:
    """Multiple hashes in consolidated file must be kept separate."""

    def test_scan_consolidated_separates_hashes(self):
        """scan_consolidated_experiments returns separate entries per hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_a = _make_cfg(label="exp_test", scale=100.0)
            cfg_b = _make_cfg(label="exp_test", scale=200.0)

            cons_path = _create_consolidated(tmpdir, [
                {"label": "exp_test", "hash": "aaaa1111", "seeds": [42, 43, 44],
                 "cfg_text": cfg_a},
                {"label": "exp_test", "hash": "bbbb2222", "seeds": [42, 43, 44],
                 "cfg_text": cfg_b},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_consolidated_experiments(cons_path, "exp_test-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 2
            assert "exp_test-aaaa1111" in exps
            assert "exp_test-bbbb2222" in exps

            # Configs should reflect different scales
            cfg_a_parsed = exps["exp_test-aaaa1111"]["config"]
            cfg_b_parsed = exps["exp_test-bbbb2222"]["config"]
            assert cfg_a_parsed["transaction"]["inter_arrival"]["scale"] == 100.0
            assert cfg_b_parsed["transaction"]["inter_arrival"]["scale"] == 200.0

    def test_consolidated_only_build_index_multi_hash(self):
        """build_experiment_index with consolidated-only multi-hash data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_a = _make_cfg(label="exp_test", scale=50.0)
            cfg_b = _make_cfg(label="exp_test", scale=150.0)

            _create_consolidated(tmpdir, [
                {"label": "exp_test", "hash": "aaaa1111", "seeds": [42, 43, 44],
                 "cfg_text": cfg_a},
                {"label": "exp_test", "hash": "bbbb2222", "seeds": [42, 43, 44],
                 "cfg_text": cfg_b},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                index_df = sa.build_experiment_index(tmpdir, "exp_test-*")
            finally:
                sa.CONFIG = old_config

            assert len(index_df) == 2
            scales = sorted(index_df["inter_arrival_scale"].tolist())
            assert scales == [50.0, 150.0]

    def test_consolidated_only_op_type_analysis(self):
        """_analyze_op_type_experiments works from consolidated-only data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(label="exp_mix", scale=100.0, fa_ratio=0.5, vo_ratio=0.5)

            _create_consolidated(tmpdir, [
                {"label": "exp_mix", "hash": "cccc3333", "seeds": [42, 43, 44],
                 "cfg_text": cfg, "n_txns": 200},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                df = sa._analyze_op_type_experiments(tmpdir, "exp_mix-*")
            finally:
                sa.CONFIG = old_config

            assert len(df) > 0, "Expected records from consolidated-only experiment"
            # Should have records for both operation types
            op_types = sorted(df["operation_type"].unique())
            assert "fast_append" in op_types
            assert "validated_overwrite" in op_types
            # Should have records for 3 seeds
            seeds = sorted(df["seed"].unique())
            assert len(seeds) == 3


# ---------------------------------------------------------------------------
# Row-group metadata scanning
# ---------------------------------------------------------------------------


class TestRowGroupMetadataScanning:
    """Verify scan_consolidated_experiments reads row-group statistics, not data."""

    def test_statistics_based_discovery(self):
        """Row-group stats provide exp_name/exp_hash/seed without reading data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(label="exp_rg", scale=100.0)
            cons_path = _create_consolidated(tmpdir, [
                {"label": "exp_rg", "hash": "rg111111", "seeds": [10, 20, 30],
                 "cfg_text": cfg},
            ])

            # Verify the file actually has statistics
            pf = pq.ParquetFile(cons_path)
            assert pf.metadata.num_row_groups == 3  # one per seed
            rg0 = pf.metadata.row_group(0)
            name_idx = pf.schema_arrow.get_field_index("exp_name")
            assert rg0.column(name_idx).is_stats_set

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_consolidated_experiments(cons_path, "exp_rg-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 1
            assert "exp_rg-rg111111" in exps
            assert sorted(exps["exp_rg-rg111111"]["seeds"]) == [10, 20, 30]

    def test_min_seeds_filter(self):
        """Experiments with fewer seeds than min_seeds are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_cfg(label="exp_few", scale=100.0)
            cons_path = _create_consolidated(tmpdir, [
                {"label": "exp_few", "hash": "fewseeds", "seeds": [42, 43],
                 "cfg_text": cfg},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            sa.CONFIG["analysis"]["min_seeds"] = 3
            try:
                exps = scan_consolidated_experiments(cons_path, "exp_few-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 0, "Should be excluded by min_seeds=3"

    def test_pattern_filtering(self):
        """Only experiments matching the pattern are returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_a = _make_cfg(label="exp_alpha", scale=100.0)
            cfg_b = _make_cfg(label="exp_beta", scale=200.0)

            cons_path = _create_consolidated(tmpdir, [
                {"label": "exp_alpha", "hash": "aaa11111", "seeds": [42, 43, 44],
                 "cfg_text": cfg_a},
                {"label": "exp_beta", "hash": "bbb22222", "seeds": [42, 43, 44],
                 "cfg_text": cfg_b},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                exps = scan_consolidated_experiments(cons_path, "exp_alpha-*")
            finally:
                sa.CONFIG = old_config

            assert len(exps) == 1
            assert "exp_alpha-aaa11111" in exps


# ---------------------------------------------------------------------------
# Consistency between disk and consolidated configs
# ---------------------------------------------------------------------------


class TestConfigConsistency:
    """Verify configs read from disk and consolidated match."""

    def test_disk_and_consolidated_configs_equivalent(self):
        """Config read from cfg.toml matches config reconstructed from consolidated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label, hsh = "exp_cfg_test", "cfg12345"
            cfg_text = _make_cfg(label=label, scale=77.0, num_tables=5, p_real=0.3)

            _create_disk_experiment(tmpdir, label, hsh, seeds=(42, 43, 44),
                                   scale=77.0, num_tables=5, p_real=0.3)
            cons_path = _create_consolidated(tmpdir, [
                {"label": label, "hash": hsh, "seeds": [42, 43, 44],
                 "cfg_text": cfg_text},
            ])

            old_config = sa.CONFIG.copy()
            sa.CONFIG = sa.get_default_config()
            try:
                disk_exps = scan_experiment_directories(tmpdir, f"{label}-*")
                cons_exps = scan_consolidated_experiments(cons_path, f"{label}-*")
            finally:
                sa.CONFIG = old_config

            disk_cfg = list(disk_exps.values())[0]["config"]
            cons_cfg = cons_exps[f"{label}-{hsh}"]["config"]

            # Key parameters should match
            for path_fn in [
                lambda c: c["transaction"]["inter_arrival"]["scale"],
                lambda c: c["catalog"]["num_tables"],
                lambda c: c["transaction"]["real_conflict_probability"],
            ]:
                assert path_fn(disk_cfg) == path_fn(cons_cfg)
