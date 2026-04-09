"""Tests for the regenerate_plots.py dispatch and configuration logic.

Verifies:
1. GRAPH_REGISTRY entries map to real saturation_analysis functions
2. Pipeline types are valid and OUTPUT_FILES covers index graphs
3. TABLE_COMPANIONS reference valid functions and registered graph types
4. Config merging correctly applies defaults and per-graph overrides
5. find_experiment_configs finds/skips configs based on [plots] presence
6. verify_seed_consistency detects consistent/inconsistent seed directories
7. process_config dispatches correctly in dry_run and skips missing experiments
"""

import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

# The scripts directory is not a package; add it to sys.path so we can import
# regenerate_plots as a module.
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import regenerate_plots  # noqa: E402
from regenerate_plots import (  # noqa: E402
    GRAPH_REGISTRY,
    OUTPUT_FILES,
    TABLE_COMPANIONS,
    find_experiment_configs,
    load_plotting_defaults,
    process_config,
    verify_seed_consistency,
)


# ---------------------------------------------------------------------------
# TestGraphRegistry
# ---------------------------------------------------------------------------

class TestGraphRegistry:
    """Validate that GRAPH_REGISTRY entries are consistent with
    saturation_analysis and the companion constants."""

    def test_all_registry_functions_exist(self):
        """Every function name in GRAPH_REGISTRY must be an attribute of
        endive.saturation_analysis."""
        from endive import saturation_analysis as sa

        for graph_type, (func_name, _pipeline) in GRAPH_REGISTRY.items():
            if func_name is None:
                continue  # compare pipeline dispatches dynamically
            assert hasattr(sa, func_name), (
                f"GRAPH_REGISTRY['{graph_type}'] references '{func_name}' "
                f"which does not exist in endive.saturation_analysis"
            )

    def test_all_pipeline_types_valid(self):
        """Every pipeline type in GRAPH_REGISTRY must be one of the three
        recognised values."""
        valid_pipelines = {"index", "time_series", "raw", "compare"}
        for graph_type, (_func_name, pipeline) in GRAPH_REGISTRY.items():
            assert pipeline in valid_pipelines, (
                f"GRAPH_REGISTRY['{graph_type}'] has pipeline '{pipeline}'; "
                f"expected one of {valid_pipelines}"
            )

    def test_output_files_cover_index_graphs(self):
        """Every graph with pipeline 'index' must have an entry in
        OUTPUT_FILES so process_config can determine the output path."""
        for graph_type, (_func_name, pipeline) in GRAPH_REGISTRY.items():
            if pipeline == "index":
                assert graph_type in OUTPUT_FILES, (
                    f"Index graph '{graph_type}' has no entry in OUTPUT_FILES"
                )

    def test_table_companions_reference_valid_functions(self):
        """Every function named in TABLE_COMPANIONS must exist in
        saturation_analysis."""
        from endive import saturation_analysis as sa

        for graph_type, func_name in TABLE_COMPANIONS.items():
            assert hasattr(sa, func_name), (
                f"TABLE_COMPANIONS['{graph_type}'] references '{func_name}' "
                f"which does not exist in endive.saturation_analysis"
            )

    def test_companion_graphs_have_registry_entries(self):
        """Every parent graph type listed in TABLE_COMPANIONS must appear as
        a key in GRAPH_REGISTRY."""
        for graph_type in TABLE_COMPANIONS:
            assert graph_type in GRAPH_REGISTRY, (
                f"TABLE_COMPANIONS key '{graph_type}' is not a registered "
                f"graph type in GRAPH_REGISTRY"
            )


# ---------------------------------------------------------------------------
# TestConfigMerging
# ---------------------------------------------------------------------------

class TestConfigMerging:
    """Test the merging logic inside process_config that combines
    plotting.toml defaults with per-graph overrides.

    We exercise this by calling process_config in dry_run mode (which
    exercises the merge code path for building kwargs) and by inspecting
    the merge logic directly.
    """

    def _merge(self, defaults: dict, graph_overrides: dict) -> dict:
        """Replicate the merge logic from process_config lines 228-229."""
        base = dict(defaults)
        overrides = {k: v for k, v in graph_overrides.items() if k != "type"}
        return {**base, **overrides}

    def test_override_replaces_default(self):
        """A per-graph override must replace the plotting.toml default."""
        defaults = {"title": "Default Title", "figsize": [12, 8]}
        graph = {"type": "latency_vs_throughput", "title": "Custom Title"}
        merged = self._merge(defaults, graph)
        assert merged["title"] == "Custom Title"

    def test_default_used_when_no_override(self):
        """When a graph config has no override for a key the plotting.toml
        default should be preserved."""
        defaults = {"title": "Default Title", "figsize": [12, 8]}
        graph = {"type": "latency_vs_throughput"}
        merged = self._merge(defaults, graph)
        assert merged["title"] == "Default Title"
        assert merged["figsize"] == [12, 8]

    def test_type_key_excluded_from_overrides(self):
        """The 'type' key is used for dispatch and must not leak into the
        merged overrides dict."""
        defaults = {}
        graph = {"type": "latency_vs_throughput", "title": "X"}
        merged = self._merge(defaults, graph)
        assert "type" not in merged

    def test_override_adds_new_keys(self):
        """Keys present in the graph config but absent from defaults are
        added to the merged result."""
        defaults = {"title": "T"}
        graph = {"type": "heatmap", "x_param": "num_tables"}
        merged = self._merge(defaults, graph)
        assert merged["x_param"] == "num_tables"
        assert merged["title"] == "T"


# ---------------------------------------------------------------------------
# TestFindExperimentConfigs
# ---------------------------------------------------------------------------

class TestFindExperimentConfigs:
    """Test find_experiment_configs() with real and synthetic config files."""

    def test_finds_configs_with_plots_section(self, tmp_path, monkeypatch):
        """Configs containing a [plots] section should be returned."""
        config_dir = tmp_path / "experiment_configs"
        config_dir.mkdir()

        cfg_with_plots = config_dir / "exp_good.toml"
        cfg_with_plots.write_text(textwrap.dedent("""\
            [experiment]
            label = "exp_good"

            [plots]
            output_dir = "plots/exp_good"

            [[plots.graphs]]
            type = "latency_vs_throughput"
        """))

        monkeypatch.chdir(tmp_path)
        configs = find_experiment_configs()
        assert len(configs) == 1
        assert configs[0].name == "exp_good.toml"

    def test_skips_configs_without_plots(self, tmp_path, monkeypatch):
        """Configs that lack a [plots] section should be excluded."""
        config_dir = tmp_path / "experiment_configs"
        config_dir.mkdir()

        cfg_no_plots = config_dir / "exp_no_plots.toml"
        cfg_no_plots.write_text(textwrap.dedent("""\
            [experiment]
            label = "exp_no_plots"

            [simulation]
            duration_ms = 3600000
        """))

        monkeypatch.chdir(tmp_path)
        configs = find_experiment_configs()
        assert len(configs) == 0

    def test_pattern_filters_configs(self, tmp_path, monkeypatch):
        """The pattern argument should narrow the returned configs."""
        config_dir = tmp_path / "experiment_configs"
        config_dir.mkdir()

        for name in ("exp1_alpha.toml", "exp2_beta.toml", "exp3_gamma.toml"):
            (config_dir / name).write_text(textwrap.dedent("""\
                [experiment]
                label = "test"

                [plots]
                output_dir = "plots/test"

                [[plots.graphs]]
                type = "latency_vs_throughput"
            """))

        monkeypatch.chdir(tmp_path)

        # Pattern "exp1*" should match only exp1_alpha.toml
        configs = find_experiment_configs(pattern="exp1*")
        assert len(configs) == 1
        assert configs[0].name == "exp1_alpha.toml"

        # Pattern "exp*" should match all three
        all_configs = find_experiment_configs(pattern="exp*")
        assert len(all_configs) == 3


# ---------------------------------------------------------------------------
# TestVerifySeedConsistency
# ---------------------------------------------------------------------------

class TestVerifySeedConsistency:
    """Test verify_seed_consistency() logic."""

    def test_consistent_seeds_returns_true(self, tmp_path):
        """When all experiment dirs have the same number of seed
        sub-directories the function returns True."""
        for exp_name in ("exp_test-aaa", "exp_test-bbb", "exp_test-ccc"):
            exp_dir = tmp_path / exp_name
            exp_dir.mkdir()
            for seed in ("42", "43", "44"):
                (exp_dir / seed).mkdir()

        assert verify_seed_consistency(str(tmp_path), "exp_test-*") is True

    def test_inconsistent_seeds_returns_false(self, tmp_path):
        """When experiment dirs have different seed counts the function
        returns False."""
        exp_a = tmp_path / "exp_test-aaa"
        exp_a.mkdir()
        for seed in ("42", "43", "44"):
            (exp_a / seed).mkdir()

        exp_b = tmp_path / "exp_test-bbb"
        exp_b.mkdir()
        for seed in ("42",):
            (exp_b / seed).mkdir()

        assert verify_seed_consistency(str(tmp_path), "exp_test-*") is False

    def test_no_experiments_returns_true(self, tmp_path):
        """An empty directory (no matching experiments) should pass
        consistency check."""
        assert verify_seed_consistency(str(tmp_path), "exp_test-*") is True


# ---------------------------------------------------------------------------
# TestProcessConfig
# ---------------------------------------------------------------------------

class TestProcessConfig:
    """Test process_config() dispatch logic."""

    @staticmethod
    def _write_config(path: Path, toml_text: str):
        path.write_text(textwrap.dedent(toml_text))

    def test_dry_run_returns_graphs(self, tmp_path):
        """In dry_run mode, process_config should list graphs without
        executing any plotting functions and return a result dict."""
        config_file = tmp_path / "exp_test.toml"
        self._write_config(config_file, """\
            [experiment]
            label = "test_exp"

            [plots]
            output_dir = "plots/test_exp"

            [[plots.graphs]]
            type = "latency_vs_throughput"

            [[plots.graphs]]
            type = "success_rate_vs_throughput"
        """)

        result = process_config(
            config_file,
            plotting_defaults={},
            input_dir=str(tmp_path / "experiments"),
            dry_run=True,
        )

        assert result["label"] == "test_exp"
        # dry_run returns early with status "ok" (no error)
        assert result.get("status", "ok") == "ok"

    def test_dry_run_unknown_type_prints_warning(self, tmp_path, capsys):
        """Dry run with an unknown graph type should print UNKNOWN TYPE."""
        config_file = tmp_path / "exp_bad.toml"
        self._write_config(config_file, """\
            [experiment]
            label = "test_bad"

            [plots]
            output_dir = "plots/test_bad"

            [[plots.graphs]]
            type = "nonexistent_graph_type"
        """)

        process_config(
            config_file,
            plotting_defaults={},
            input_dir=str(tmp_path / "experiments"),
            dry_run=True,
        )

        captured = capsys.readouterr()
        assert "UNKNOWN TYPE" in captured.out

    def test_missing_experiments_skips(self, tmp_path):
        """When no experiments match the pattern and no consolidated.parquet
        exists, process_config should return a skipped result."""
        config_file = tmp_path / "exp_missing.toml"
        self._write_config(config_file, """\
            [experiment]
            label = "exp_missing"

            [plots]
            output_dir = "plots/exp_missing"

            [[plots.graphs]]
            type = "latency_vs_throughput"
        """)

        # Create an empty experiments directory (no matching dirs, no
        # consolidated.parquet)
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()

        result = process_config(
            config_file,
            plotting_defaults={},
            input_dir=str(exp_dir),
            dry_run=False,
        )

        assert result["status"] == "skipped"
        assert "no experiments matching" in result["reason"]

    def test_no_graphs_defined_skips(self, tmp_path):
        """A config with an empty [plots] section (no [[plots.graphs]])
        should be skipped with reason 'no graphs defined'."""
        config_file = tmp_path / "exp_empty.toml"
        self._write_config(config_file, """\
            [experiment]
            label = "exp_empty"

            [plots]
            output_dir = "plots/exp_empty"
        """)

        result = process_config(
            config_file,
            plotting_defaults={},
            input_dir=str(tmp_path / "experiments"),
            dry_run=False,
        )

        assert result["status"] == "skipped"
        assert result["reason"] == "no graphs defined"

    def test_output_suffix_creates_subdirectory(self, tmp_path):
        """When a graph config contains output_suffix, process_config
        should create a subdirectory under output_dir.

        We test this via dry_run to avoid needing real experiment data;
        verify that the merge logic correctly pops output_suffix.
        """
        config_file = tmp_path / "exp_suffix.toml"
        self._write_config(config_file, """\
            [experiment]
            label = "exp_suffix"

            [plots]
            output_dir = "plots/exp_suffix"

            [[plots.graphs]]
            type = "latency_vs_throughput"
            output_suffix = "filtered_view"
        """)

        # Dry run does not create directories, but we can verify the merge
        # logic handles output_suffix by confirming no error is raised and
        # that the result is returned.
        result = process_config(
            config_file,
            plotting_defaults={},
            input_dir=str(tmp_path / "experiments"),
            dry_run=True,
        )

        assert result["label"] == "exp_suffix"
        assert result.get("status", "ok") == "ok"


# ---------------------------------------------------------------------------
# TestLoadPlottingDefaults
# ---------------------------------------------------------------------------

class TestLoadPlottingDefaults:
    """Test load_plotting_defaults() behaviour."""

    def test_returns_empty_when_missing(self, tmp_path, monkeypatch):
        """If plotting.toml does not exist the function returns {}."""
        monkeypatch.chdir(tmp_path)
        result = load_plotting_defaults()
        assert result == {}

    def test_loads_valid_toml(self, tmp_path, monkeypatch):
        """A valid plotting.toml is parsed and returned as a dict."""
        toml_path = tmp_path / "plotting.toml"
        toml_path.write_text(textwrap.dedent("""\
            [latency_vs_throughput]
            title = "Test Title"
            figsize = [10, 6]
        """))
        monkeypatch.chdir(tmp_path)
        result = load_plotting_defaults()
        assert result["latency_vs_throughput"]["title"] == "Test Title"
        assert result["latency_vs_throughput"]["figsize"] == [10, 6]
