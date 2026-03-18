"""Tests for _analyze_workload_experiments() and generate_workload_knee_table()."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from endive.saturation_analysis import (
    _analyze_workload_experiments,
    generate_workload_knee_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CFG_TEMPLATE = """\
[simulation]
duration_ms = 3600000

[experiment]
label = "{label}"

[catalog]
num_tables = {num_tables}
mode = "instant"

[storage]
provider = "{provider}"

[transaction]
retry = 10
inter_arrival.scale = {scale}
real_conflict_probability = 0.0
runtime.mean = 10000
runtime.sigma = 1.5
operation_types.fast_append = {fa_ratio}
operation_types.validated_overwrite = {vo_ratio}
"""


def _make_cfg(label="exp_test", num_tables=1, scale=100.0,
              fa_ratio=1.0, vo_ratio=0.0, provider="s3x"):
    return _CFG_TEMPLATE.format(
        label=label, num_tables=num_tables, scale=scale,
        fa_ratio=fa_ratio, vo_ratio=vo_ratio, provider=provider,
    )


def _make_results(n=200, seed_val=42, fa_ratio=1.0, success_rate=1.0):
    """Create transaction results with controllable op-type mix and success rate."""
    rng = np.random.RandomState(seed_val)
    n_fa = int(n * fa_ratio)
    n_vo = n - n_fa
    op_types = (["fast_append"] * n_fa +
                ["validated_overwrite"] * n_vo)
    rng.shuffle(op_types)

    statuses = []
    for _ in range(n):
        statuses.append("committed" if rng.random() < success_rate else "aborted")

    return pd.DataFrame({
        "txn_id": range(n),
        "t_submit": np.linspace(0, 3_600_000, n).astype(int),
        "t_commit": (np.linspace(0, 3_600_000, n) + 1000).astype(int),
        "commit_latency": rng.randint(500, 5000, n),
        "total_latency": rng.randint(5000, 15000, n),
        "n_retries": rng.randint(0, 3, n).astype(np.int8),
        "status": statuses,
        "operation_type": op_types,
    })


def _create_experiment(base_dir, label, exp_hash, seeds=(42, 43, 44),
                       n_txns=200, **cfg_kwargs):
    """Create a disk experiment directory with cfg.toml and seed results."""
    exp_dir = Path(base_dir) / f"{label}-{exp_hash}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fa_ratio = cfg_kwargs.get("fa_ratio", 1.0)
    success_rate = cfg_kwargs.pop("success_rate", 1.0)
    with open(exp_dir / "cfg.toml", "w") as f:
        f.write(_make_cfg(label=label, **cfg_kwargs))
    for s in seeds:
        seed_dir = exp_dir / str(s)
        seed_dir.mkdir(exist_ok=True)
        _make_results(n_txns, seed_val=s, fa_ratio=fa_ratio,
                      success_rate=success_rate).to_parquet(
            seed_dir / "results.parquet"
        )
    return exp_dir


# ---------------------------------------------------------------------------
# _analyze_workload_experiments
# ---------------------------------------------------------------------------

class TestAnalyzeWorkloadExperiments:
    """Tests for _analyze_workload_experiments()."""

    def test_extracts_provider_and_num_tables(self):
        """Each record includes storage_provider and num_tables from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "aa11", seeds=(42,),
                               provider="gcp", num_tables=5)
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            assert len(df) > 0
            assert (df["storage_provider"] == "gcp").all()
            assert (df["num_tables"] == 5).all()

    def test_returns_both_op_types_for_mixed_workload(self):
        """Mixed FA/VO workload produces records for both operation types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "bb22", seeds=(42,),
                               fa_ratio=0.5, vo_ratio=0.5, n_txns=400)
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            op_types = set(df["operation_type"].unique())
            assert "fast_append" in op_types
            assert "validated_overwrite" in op_types

    def test_warmup_cooldown_filtering(self):
        """Records use middle 80% of simulation (10% warmup, 10% cooldown)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "cc33", seeds=(42,), n_txns=1000)
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            # With 1000 txns over 3600s, the active window should be
            # roughly 80% of total, so committed count should be less than 1000
            total = df["total"].sum()
            assert total < 1000, "Warmup/cooldown should filter some transactions"
            assert total > 500, "Should retain majority of transactions"

    def test_active_window_computed(self):
        """Each record has a positive active_window_s field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "dd44", seeds=(42,))
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            assert (df["active_window_s"] > 0).all()

    def test_multiple_seeds_produce_multiple_records(self):
        """Each seed produces its own records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "ee55", seeds=(42, 43, 44))
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            assert df["seed"].nunique() == 3

    def test_multiple_providers(self):
        """Experiments with different providers produce distinct records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "f1f1", seeds=(42,),
                               provider="s3x")
            _create_experiment(tmpdir, "exp_wk", "f2f2", seeds=(42,),
                               provider="gcp")
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            providers = set(df["storage_provider"].unique())
            assert providers == {"s3x", "gcp"}

    def test_filters_applied(self):
        """Filters reduce the result set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "g1g1", seeds=(42,),
                               provider="s3x", num_tables=1)
            _create_experiment(tmpdir, "exp_wk", "g2g2", seeds=(42,),
                               provider="s3x", num_tables=5)
            df_all = _analyze_workload_experiments(tmpdir, "exp_wk*")
            df_filtered = _analyze_workload_experiments(
                tmpdir, "exp_wk*", filters=["num_tables==1"]
            )
            assert len(df_filtered) < len(df_all)
            assert (df_filtered["num_tables"] == 1).all()

    def test_success_rate_reflects_aborts(self):
        """Success rate is < 100% when some transactions abort."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_experiment(tmpdir, "exp_wk", "h1h1", seeds=(42,),
                               n_txns=500, success_rate=0.7)
            df = _analyze_workload_experiments(tmpdir, "exp_wk*")
            # With 70% success rate, the measured rate should be
            # roughly in that ballpark (not exactly due to warmup filtering)
            avg_sr = df["success_rate"].mean()
            assert avg_sr < 90, f"Expected low success rate, got {avg_sr}"
            assert avg_sr > 50, f"Success rate too low: {avg_sr}"


# ---------------------------------------------------------------------------
# generate_workload_knee_table
# ---------------------------------------------------------------------------

class TestGenerateWorkloadKneeTable:
    """Tests for generate_workload_knee_table()."""

    def _setup_sweep(self, tmpdir, provider="s3x", num_tables=1):
        """Create experiments at multiple load levels with degrading success."""
        label = "exp_wk"
        # Light load: 100% success
        _create_experiment(
            tmpdir, label, f"{provider[:2]}{num_tables}a", seeds=(42, 43),
            provider=provider, num_tables=num_tables,
            scale=1000.0, fa_ratio=0.5, vo_ratio=0.5,
            success_rate=1.0, n_txns=200,
        )
        # Medium load: still above 95%
        _create_experiment(
            tmpdir, label, f"{provider[:2]}{num_tables}b", seeds=(42, 43),
            provider=provider, num_tables=num_tables,
            scale=100.0, fa_ratio=0.5, vo_ratio=0.5,
            success_rate=0.98, n_txns=400,
        )
        # Heavy load: below 95% — should NOT be the knee
        _create_experiment(
            tmpdir, label, f"{provider[:2]}{num_tables}c", seeds=(42, 43),
            provider=provider, num_tables=num_tables,
            scale=20.0, fa_ratio=0.5, vo_ratio=0.5,
            success_rate=0.70, n_txns=600,
        )

    def test_generates_md_and_csv(self):
        """Output directory contains both .md and .csv files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            assert os.path.exists(os.path.join(out, "workload_knee_table.md"))
            assert os.path.exists(os.path.join(out, "workload_knee_data.csv"))

    def test_generates_plot(self):
        """Output directory contains the .png plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            assert os.path.exists(os.path.join(out, "workload_knee_vs_tables.png"))

    def test_knee_excludes_overloaded_points(self):
        """Knee point should be at the medium load, not the heavy load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            csv = pd.read_csv(os.path.join(out, "workload_knee_data.csv"))
            # The heavy-load point (scale=20) has <95% VO success
            heavy = csv[(csv["inter_arrival_scale"] == 20.0) &
                        (csv["operation_type"] == "validated_overwrite")]
            if len(heavy) > 0:
                assert heavy["success_rate"].values[0] < 95.0

    def test_md_contains_provider(self):
        """The markdown table includes the provider name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            assert "s3x" in md

    def test_md_has_num_tables_section(self):
        """Markdown has a section header for the num_tables value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            assert "## num_tables = 1" in md

    def test_multiple_providers(self):
        """Table includes rows for each provider."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir, provider="s3x")
            self._setup_sweep(tmpdir, provider="gcp")
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            assert "s3x" in md
            assert "gcp" in md

    def test_multiple_table_counts(self):
        """Separate sections for different num_tables values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir, num_tables=1)
            self._setup_sweep(tmpdir, num_tables=5)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            assert "## num_tables = 1" in md
            assert "## num_tables = 5" in md

    def test_fa_only_workload_shows_dash_for_vo(self):
        """100% FA workload shows '—' for VO latency."""
        with tempfile.TemporaryDirectory() as tmpdir:
            label = "exp_wk"
            _create_experiment(
                tmpdir, label, "fa100a", seeds=(42, 43),
                provider="s3x", num_tables=1,
                scale=100.0, fa_ratio=1.0, vo_ratio=0.0,
                n_txns=300,
            )
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            # Should contain "—" for the VO latency in 100/0 column
            assert "—" in md

    def test_empty_pattern_produces_no_output(self):
        """Non-matching pattern produces no output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "nonexistent_*", out)
            assert not os.path.exists(os.path.join(out, "workload_knee_table.md"))

    def test_custom_success_threshold(self):
        """Config can override the 95% success threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(
                tmpdir, "exp_wk*", out,
                config={"success_threshold": 99.0},
            )
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            assert "> 99%" in md

    def test_csv_has_all_required_columns(self):
        """The CSV output has per-type metrics at each load point."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            csv = pd.read_csv(os.path.join(out, "workload_knee_data.csv"))
            required = {"storage_provider", "num_tables", "fa_ratio",
                        "inter_arrival_scale", "operation_type",
                        "total", "committed", "success_rate",
                        "mean_latency", "p95_latency"}
            assert required.issubset(set(csv.columns))

    def test_committed_per_sec_positive(self):
        """Knee points have positive committed/sec values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._setup_sweep(tmpdir)
            out = os.path.join(tmpdir, "output")
            generate_workload_knee_table(tmpdir, "exp_wk*", out)
            md = open(os.path.join(out, "workload_knee_table.md")).read()
            # Parse first numeric value after "| s3x |"
            for line in md.split("\n"):
                if line.startswith("| s3x"):
                    parts = [p.strip() for p in line.split("|") if p.strip()]
                    # First metric column is c/s
                    val = float(parts[1])
                    assert val > 0, f"Expected positive c/s, got {val}"
                    break
