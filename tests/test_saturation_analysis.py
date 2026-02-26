"""Tests for saturation analysis module.

Verifies that:
1. Experiment directories are scanned correctly
2. Parameters are extracted from cfg.toml files
3. Results are loaded and aggregated across seeds
4. Statistics are computed correctly
5. Index is built with correct structure
6. Plots are generated successfully
"""

import os
import tempfile
import pytest
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from endive.saturation_analysis import (
    scan_experiment_directories,
    extract_key_parameters,
    load_and_aggregate_results,
    compute_aggregate_statistics,
    build_experiment_index,
    save_experiment_index,
    get_default_config,
    load_config,
    compute_transient_period_duration
)
import endive.saturation_analysis as saturation_analysis


class TestExperimentScanning:
    """Test experiment directory scanning functionality."""

    def test_scan_finds_experiments(self):
        """Verify scan finds experiment directories with cfg.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test experiment structure
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            # Create cfg.toml
            cfg_content = """
[simulation]
duration_ms = 1000
output_path = "results.parquet"

[experiment]
label = "exp_test"

[catalog]
num_tables = 1

[transaction]
inter_arrival.scale = 100
real_conflict_probability = 0.0
"""
            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write(cfg_content)

            # Create seed directory
            seed_dir = exp_dir / "12345"
            seed_dir.mkdir()

            # Scan
            experiments = scan_experiment_directories(tmpdir, "exp_test-*")

            assert len(experiments) == 1
            exp_info = experiments[str(exp_dir)]
            assert exp_info['label'] == "exp_test"
            assert exp_info['hash'] == "abc12345"
            assert len(exp_info['seeds']) == 1

            print("✓ Experiment scanning works")

    def test_scan_skips_without_cfg(self):
        """Verify scan skips directories without cfg.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory without cfg.toml
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            # Scan should find nothing
            experiments = scan_experiment_directories(tmpdir, "exp_test-*")
            assert len(experiments) == 0

            print("✓ Skips directories without cfg.toml")

    def test_scan_skips_without_seeds(self):
        """Verify scan skips experiments without seed directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            # Create cfg.toml but no seed directories
            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write("[simulation]\nduration_ms = 1000\n")

            # Scan should find nothing (no seeds)
            experiments = scan_experiment_directories(tmpdir, "exp_test-*")
            assert len(experiments) == 0

            print("✓ Skips experiments without seed directories")

    def test_scan_finds_multiple_seeds(self):
        """Verify scan finds all seed directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write("[simulation]\nduration_ms = 1000\n")

            # Create multiple seed directories
            for seed in [111, 222, 333]:
                (exp_dir / str(seed)).mkdir()

            experiments = scan_experiment_directories(tmpdir, "exp_test-*")

            assert len(experiments) == 1
            exp_info = experiments[str(exp_dir)]
            assert len(exp_info['seeds']) == 3

            print("✓ Finds multiple seed directories")


class TestParameterExtraction:
    """Test parameter extraction from config."""

    def test_extract_inter_arrival(self):
        """Verify inter_arrival parameter extraction."""
        config = {
            'transaction': {
                'inter_arrival': {
                    'scale': 100.0
                }
            }
        }

        params = extract_key_parameters(config)
        assert params['inter_arrival_scale'] == 100.0

        print("✓ Extracts inter_arrival.scale")

    def test_extract_catalog_params(self):
        """Verify catalog parameter extraction."""
        config = {
            'catalog': {
                'num_tables': 10
            }
        }

        params = extract_key_parameters(config)
        assert params['num_tables'] == 10

        print("✓ Extracts catalog parameters")

    def test_extract_conflict_probability(self):
        """Verify conflict probability extraction."""
        config = {
            'transaction': {
                'real_conflict_probability': 0.5
            }
        }

        params = extract_key_parameters(config)
        assert params['real_conflict_probability'] == 0.5

        print("✓ Extracts real_conflict_probability")

    def test_extract_handles_missing_params(self):
        """Verify extraction handles missing parameters gracefully."""
        config = {}

        params = extract_key_parameters(config)

        # Should not raise error, just return empty or None values
        assert isinstance(params, dict)

        print("✓ Handles missing parameters")


class TestResultsLoading:
    """Test loading and aggregating results."""

    def test_load_single_seed(self):
        """Verify loading results from single seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test parquet file
            seed_dir = Path(tmpdir) / "12345"
            seed_dir.mkdir()

            # Create sample results spanning 1 hour
            n_txns = 60
            df = pd.DataFrame({
                'txn_id': range(n_txns),
                't_submit': [i * 60000 for i in range(n_txns)],
                't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                'commit_latency': [1000] * n_txns,
                'total_latency': [11000] * n_txns,
                'n_retries': [0] * n_txns,
                'status': ['committed'] * n_txns
            })
            df.to_parquet(seed_dir / "results.parquet")

            # Load
            exp_info = {
                'seeds': [str(seed_dir)],
                'config': {
                    'transaction': {'runtime': {'mean': 10000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }
            result = load_and_aggregate_results(exp_info)

            assert result is not None
            assert len(result) > 0  # Some transactions after warmup/cooldown filtering
            assert 'seed' in result.columns
            assert result['seed'].iloc[0] == "12345"

            print("✓ Loads single seed results")

    def test_aggregate_multiple_seeds(self):
        """Verify aggregation across multiple seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_dirs = []

            # Create results for 3 seeds
            for seed in [111, 222, 333]:
                seed_dir = Path(tmpdir) / str(seed)
                seed_dir.mkdir()
                seed_dirs.append(str(seed_dir))

                n_txns = 60
                df = pd.DataFrame({
                    'txn_id': range(n_txns),
                    't_submit': [i * 60000 for i in range(n_txns)],
                    't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                    'commit_latency': [1000] * n_txns,
                    'total_latency': [11000] * n_txns,
                    'n_retries': [0] * n_txns,
                    'status': ['committed'] * n_txns
                })
                df.to_parquet(seed_dir / "results.parquet")

            # Aggregate
            exp_info = {
                'seeds': seed_dirs,
                'config': {
                    'transaction': {'runtime': {'mean': 10000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }
            result = load_and_aggregate_results(exp_info)

            assert result is not None
            assert len(result) > 0  # Some transactions after warmup/cooldown filtering

            # Check all seeds present
            assert set(result['seed'].unique()) == {'111', '222', '333'}

            print("✓ Aggregates multiple seeds")


class TestStatisticsComputation:
    """Test aggregate statistics computation."""

    def test_compute_basic_stats(self):
        """Verify basic statistics computation."""
        df = pd.DataFrame({
            't_submit': [0, 100, 200, 300, 400],
            'commit_latency': [100, 200, 150, 250, 300],
            'total_latency': [200, 300, 250, 350, 400],
            'n_retries': [0, 1, 0, 2, 1],
            'status': ['committed'] * 5
        })

        stats = compute_aggregate_statistics(df)

        assert stats is not None
        assert stats['total_txns'] == 5
        assert stats['committed'] == 5
        assert stats['success_rate'] == 100.0
        assert stats['mean_commit_latency'] == 200.0
        assert stats['p50_commit_latency'] == 200.0

        print("✓ Computes basic statistics")

    def test_compute_with_failures(self):
        """Verify statistics with failed transactions."""
        df = pd.DataFrame({
            't_submit': [0, 100, 200, 300, 400],
            'commit_latency': [100, 200, 150, 250, 300],
            'total_latency': [200, 300, 250, 350, 400],
            'n_retries': [0, 1, 0, 2, 1],
            'status': ['committed', 'committed', 'aborted', 'committed', 'aborted']
        })

        stats = compute_aggregate_statistics(df)

        assert stats['total_txns'] == 5
        assert stats['committed'] == 3
        assert stats['aborted'] == 2
        assert stats['success_rate'] == 60.0

        print("✓ Handles failed transactions")

    def test_compute_throughput(self):
        """Verify throughput computation."""
        df = pd.DataFrame({
            't_submit': [0, 1000, 2000, 3000, 4000],  # Over 4 seconds
            'commit_latency': [100] * 5,
            'total_latency': [200] * 5,
            'n_retries': [0] * 5,
            'status': ['committed'] * 5
        })

        stats = compute_aggregate_statistics(df)

        # 5 commits over 4 seconds = 1.25 commits/sec
        assert abs(stats['throughput'] - 1.25) < 0.01

        print("✓ Computes throughput correctly")

    def test_compute_percentiles(self):
        """Verify latency percentile computation."""
        # Create data with known percentiles
        latencies = list(range(1, 101))  # 1 to 100
        df = pd.DataFrame({
            't_submit': range(100),
            'commit_latency': latencies,
            'total_latency': [l + 100 for l in latencies],
            'n_retries': [0] * 100,
            'status': ['committed'] * 100
        })

        stats = compute_aggregate_statistics(df)

        # For uniform 1-100, p50 should be ~50, p95 ~95, p99 ~99
        assert 48 <= stats['p50_commit_latency'] <= 52
        assert 93 <= stats['p95_commit_latency'] <= 97
        assert 97 <= stats['p99_commit_latency'] <= 100

        print("✓ Computes percentiles correctly")


class TestIndexBuilding:
    """Test experiment index building."""

    def test_build_index_structure(self):
        """Verify index has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal experiment
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            cfg_content = """
[simulation]
duration_ms = 3600000

[experiment]
label = "exp_test"

[catalog]
num_tables = 1

[transaction]
inter_arrival.scale = 100
real_conflict_probability = 0.0
runtime.mean = 10000
"""
            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write(cfg_content)

            seed_dir = exp_dir / "12345"
            seed_dir.mkdir()

            # Create sample results spanning 1 hour
            n_txns = 60
            df = pd.DataFrame({
                'txn_id': range(n_txns),
                't_submit': [i * 60000 for i in range(n_txns)],
                't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                'commit_latency': [1000] * n_txns,
                'total_latency': [11000] * n_txns,
                'n_retries': [0] * n_txns,
                'status': ['committed'] * n_txns,
                't_runtime': [10000] * n_txns
            })
            df.to_parquet(seed_dir / "results.parquet")

            # Build index with min_seeds=1 for single-seed test
            original_config = saturation_analysis.CONFIG.copy()
            try:
                saturation_analysis.CONFIG['analysis'] = saturation_analysis.CONFIG.get('analysis', {}).copy()
                saturation_analysis.CONFIG['analysis']['min_seeds'] = 1
                index_df = build_experiment_index(tmpdir, "exp_test-*")
            finally:
                saturation_analysis.CONFIG = original_config

            assert len(index_df) == 1
            assert 'label' in index_df.columns
            assert 'hash' in index_df.columns
            assert 'inter_arrival_scale' in index_df.columns
            assert 'success_rate' in index_df.columns
            assert 'throughput' in index_df.columns

            print("✓ Index has correct structure")

    def test_build_index_aggregates_seeds(self):
        """Verify index aggregates across seeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / "exp_test-abc12345"
            exp_dir.mkdir()

            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write("[simulation]\nduration_ms = 3600000\n[experiment]\nlabel = \"exp_test\"\n[transaction]\ninter_arrival.scale = 100\nruntime.mean = 10000\n")

            # Create 3 seeds with results spanning 1 hour
            for seed in [111, 222, 333]:
                seed_dir = exp_dir / str(seed)
                seed_dir.mkdir()

                n_txns = 60
                df = pd.DataFrame({
                    'txn_id': range(n_txns),
                    't_submit': [i * 60000 for i in range(n_txns)],
                    't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                    'commit_latency': [1000] * n_txns,
                    'total_latency': [11000] * n_txns,
                    'n_retries': [0] * n_txns,
                    'status': ['committed'] * n_txns,
                    't_runtime': [10000] * n_txns
                })
                df.to_parquet(seed_dir / "results.parquet")

            index_df = build_experiment_index(tmpdir, "exp_test-*")

            assert len(index_df) == 1
            row = index_df.iloc[0]

            # Should aggregate: 60 txns × 3 seeds = 180 total (but some excluded by warmup/cooldown)
            assert row['num_seeds'] == 3
            # After warmup/cooldown exclusion, should have some transactions
            assert row['total_txns'] > 0

            print("✓ Index aggregates across seeds")


class TestIndexExport:
    """Test experiment index export."""

    def test_export_to_csv(self):
        """Verify index exports to CSV correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample index
            index_df = pd.DataFrame({
                'label': ['exp_test'],
                'hash': ['abc12345'],
                'num_seeds': [3],
                'inter_arrival_scale': [100.0],
                'num_tables': [1],
                'success_rate': [95.5],
                'throughput': [10.5],
                'mean_commit_latency': [200.0]
            })

            output_path = Path(tmpdir) / "index.csv"
            save_experiment_index(index_df, str(output_path))

            # Verify file exists and can be read
            assert output_path.exists()
            loaded = pd.read_csv(output_path)

            assert len(loaded) == 1
            assert loaded['label'].iloc[0] == 'exp_test'
            assert loaded['hash'].iloc[0] == 'abc12345'
            assert abs(loaded['success_rate'].iloc[0] - 95.5) < 0.1

            print("✓ Exports to CSV correctly")


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self):
        """Verify complete analysis workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment with multiple parameter combinations
            for ia_scale in [100, 500]:
                for seed in [111, 222]:
                    # Create unique hash for each parameter combination
                    exp_hash = f"hash{ia_scale}"
                    exp_dir = Path(tmpdir) / f"exp_test-{exp_hash}"
                    exp_dir.mkdir(exist_ok=True)

                    # Only write cfg.toml once per experiment
                    cfg_path = exp_dir / "cfg.toml"
                    if not cfg_path.exists():
                        with open(cfg_path, 'w') as f:
                            f.write(f"""
[simulation]
duration_ms = 3600000

[experiment]
label = "exp_test"

[catalog]
num_tables = 1

[transaction]
inter_arrival.scale = {ia_scale}
real_conflict_probability = 0.0
runtime.mean = 10000
""")

                    # Create seed directory
                    seed_dir = exp_dir / str(seed)
                    seed_dir.mkdir()

                    # Create results spanning 1 hour
                    n_txns = 60
                    df = pd.DataFrame({
                        'txn_id': range(n_txns),
                        't_submit': [i * 60000 for i in range(n_txns)],
                        't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                        'commit_latency': [1000] * n_txns,
                        'total_latency': [11000] * n_txns,
                        'n_retries': [0] * n_txns,
                        'status': ['committed'] * n_txns,
                        't_runtime': [10000] * n_txns
                    })
                    df.to_parquet(seed_dir / "results.parquet")

            # Build index with min_seeds=2 for two-seed test
            original_config = saturation_analysis.CONFIG.copy()
            try:
                saturation_analysis.CONFIG['analysis'] = saturation_analysis.CONFIG.get('analysis', {}).copy()
                saturation_analysis.CONFIG['analysis']['min_seeds'] = 2
                index_df = build_experiment_index(tmpdir, "exp_test-*")
            finally:
                saturation_analysis.CONFIG = original_config

            # Should have 2 experiments (one per inter_arrival value)
            assert len(index_df) == 2

            # Verify parameter extraction
            assert 100 in index_df['inter_arrival_scale'].values
            assert 500 in index_df['inter_arrival_scale'].values

            # Both experiments should have statistics
            assert 'throughput' in index_df.columns
            assert 'success_rate' in index_df.columns

            print("✓ Full workflow works correctly")


class TestConfigurationSystem:
    """Test configuration loading and CLI overrides."""

    def test_default_config_structure(self):
        """Verify default config has all required sections."""
        config = get_default_config()

        # Check top-level sections
        assert 'paths' in config
        assert 'analysis' in config
        assert 'plots' in config
        assert 'output' in config

        # Check paths section
        assert config['paths']['input_dir'] == 'experiments'
        assert config['paths']['output_dir'] == 'plots'
        assert config['paths']['pattern'] == 'exp2_*'

        # Check analysis section
        assert config['analysis']['k_min_cycles'] == 5
        assert config['analysis']['min_warmup_ms'] == 300000
        assert config['analysis']['max_warmup_ms'] == 900000
        assert config['analysis']['group_by'] is None

        # Check plots section
        assert config['plots']['dpi'] == saturation_analysis.DEFAULT_DPI
        assert 'figsize' in config['plots']
        assert 'fonts' in config['plots']
        assert 'styles' in config['plots']

        # Check output section
        assert 'files' in config['output']
        assert 'table' in config['output']

        print("✓ Default config has correct structure")

    def test_load_config_from_file(self):
        """Verify config loads from TOML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_analysis.toml"

            # Create test config
            config_content = """
[paths]
input_dir = "test_experiments"
output_dir = "test_plots"
pattern = "test_exp_*"

[analysis]
k_min_cycles = 10
min_warmup_ms = 600000
max_warmup_ms = 1800000
group_by = "num_tables"

[plots]
dpi = 150

[output.files]
experiment_index = "custom_index.csv"
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            # Load config
            config = load_config(str(config_path))

            # Verify loaded values
            assert config['paths']['input_dir'] == 'test_experiments'
            assert config['paths']['output_dir'] == 'test_plots'
            assert config['paths']['pattern'] == 'test_exp_*'
            assert config['analysis']['k_min_cycles'] == 10
            assert config['analysis']['min_warmup_ms'] == 600000
            assert config['analysis']['group_by'] == 'num_tables'
            assert config['plots']['dpi'] == 150
            assert config['output']['files']['experiment_index'] == 'custom_index.csv'

            print("✓ Loads config from file")

    def test_load_config_partial_override(self):
        """Verify partial config merges with defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "partial.toml"

            # Create partial config (only override paths)
            config_content = """
[paths]
input_dir = "custom_experiments"
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = load_config(str(config_path))

            # Overridden value
            assert config['paths']['input_dir'] == 'custom_experiments'

            # Default values should still be present
            assert config['paths']['output_dir'] == 'plots'
            assert config['analysis']['k_min_cycles'] == 5
            assert config['plots']['dpi'] == saturation_analysis.DEFAULT_DPI
            assert 'experiment_index' in config['output']['files']

            print("✓ Partial config merges with defaults")

    def test_load_config_file_not_found(self):
        """Verify graceful handling of missing config file."""
        config = load_config("nonexistent_config.toml")

        # Should return default config
        assert config == get_default_config()

        print("✓ Handles missing config file gracefully")

    def test_load_config_malformed_toml(self):
        """Verify graceful handling of malformed TOML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "malformed.toml"

            # Create malformed TOML
            with open(config_path, 'w') as f:
                f.write("[paths\ninvalid toml syntax")

            config = load_config(str(config_path))

            # Should return default config
            assert config == get_default_config()

            print("✓ Handles malformed TOML gracefully")

    def test_load_config_deep_merge(self):
        """Verify deep merge preserves nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "deep_merge.toml"

            # Override only one nested value
            config_content = """
[plots.fonts]
title = 20
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = load_config(str(config_path))

            # Overridden value
            assert config['plots']['fonts']['title'] == 20

            # Other font values should still have defaults
            assert config['plots']['fonts']['axis_label'] == 14
            assert config['plots']['fonts']['legend'] == 10

            # Other plot values should be present
            assert config['plots']['dpi'] == saturation_analysis.DEFAULT_DPI
            assert 'figsize' in config['plots']

            print("✓ Deep merge preserves nested structures")

    def test_load_config_auto_discovery(self):
        """Verify automatic discovery of analysis.toml in current directory."""
        original_dir = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)

                # Create analysis.toml in current directory
                config_content = """
[paths]
input_dir = "auto_discovered"
"""
                with open("analysis.toml", 'w') as f:
                    f.write(config_content)

                # Load without specifying path
                config = load_config()

                assert config['paths']['input_dir'] == 'auto_discovered'

                print("✓ Auto-discovers analysis.toml")
        finally:
            os.chdir(original_dir)

    def test_config_used_in_warmup_calculation(self):
        """Verify CONFIG values are used in compute_transient_period_duration()."""
        # Save original config
        original_config = saturation_analysis.CONFIG.copy()

        try:
            # Set custom config values
            saturation_analysis.CONFIG = {
                'analysis': {
                    'k_min_cycles': 10,  # Higher than default 5
                    'min_warmup_ms': 600000,  # 10 minutes
                    'max_warmup_ms': 1800000   # 30 minutes
                }
            }

            # Test with mean runtime of 60 seconds
            config = {
                'transaction': {
                    'runtime': {
                        'mean': 60000  # 60 seconds
                    }
                }
            }

            warmup = compute_transient_period_duration(config)

            # With k=10 and mean runtime of 60s, computed warmup = 10 * 60s = 600s = 600000ms
            # This should be clamped by min_warmup_ms (600000ms)
            assert warmup == 600000

            # Test with very short runtime to verify min_warmup_ms is enforced
            config_short = {
                'transaction': {
                    'runtime': {
                        'mean': 1000  # 1 second
                    }
                }
            }

            warmup_short = compute_transient_period_duration(config_short)
            # k=10 * 1s = 10s = 10000ms, but min is 600000ms
            assert warmup_short == 600000

            # Test with very long runtime to verify max_warmup_ms is enforced
            config_long = {
                'transaction': {
                    'runtime': {
                        'mean': 300000  # 5 minutes
                    }
                }
            }

            warmup_long = compute_transient_period_duration(config_long)
            # k=10 * 300s = 3000s = 3000000ms, but max is 1800000ms
            assert warmup_long == 1800000

            print("✓ CONFIG values used in warmup calculation")
        finally:
            # Restore original config
            saturation_analysis.CONFIG = original_config

    def test_output_filenames_from_config(self):
        """Verify output filenames are read from CONFIG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "custom_names.toml"

            # Create config with custom filenames
            config_content = """
[output.files]
experiment_index = "my_index.csv"
latency_vs_throughput_plot = "my_latency.png"
latency_vs_throughput_table = "my_latency.md"
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = load_config(str(config_path))

            # Verify custom filenames are loaded
            assert config['output']['files']['experiment_index'] == 'my_index.csv'
            assert config['output']['files']['latency_vs_throughput_plot'] == 'my_latency.png'
            assert config['output']['files']['latency_vs_throughput_table'] == 'my_latency.md'

            # Default filenames should still be present for non-overridden values
            assert config['output']['files']['success_vs_load_plot'] == 'success_vs_load.png'

            print("✓ Output filenames read from config")

    def test_all_default_keys_present(self):
        """Verify all expected config keys are present in defaults."""
        config = get_default_config()

        # Paths
        assert 'input_dir' in config['paths']
        assert 'output_dir' in config['paths']
        assert 'pattern' in config['paths']

        # Analysis
        assert 'group_by' in config['analysis']
        assert 'k_min_cycles' in config['analysis']
        assert 'min_warmup_ms' in config['analysis']
        assert 'max_warmup_ms' in config['analysis']

        # Plots
        assert 'dpi' in config['plots']
        assert 'bbox_inches' in config['plots']
        assert 'figsize' in config['plots']
        assert 'fonts' in config['plots']
        assert 'font_weights' in config['plots']
        assert 'legend' in config['plots']
        assert 'annotation' in config['plots']
        assert 'styles' in config['plots']
        assert 'grid' in config['plots']
        assert 'percentiles' in config['plots']
        assert 'colors' in config['plots']

        # Plot subsections
        assert 'latency_vs_throughput' in config['plots']['figsize']
        assert 'success_vs_load' in config['plots']['figsize']
        assert 'success_vs_throughput' in config['plots']['figsize']
        assert 'overhead_vs_throughput' in config['plots']['figsize']

        assert 'title' in config['plots']['fonts']
        assert 'axis_label' in config['plots']['fonts']
        assert 'legend' in config['plots']['fonts']

        assert 'markers' in config['plots']['styles']
        assert 'linestyles' in config['plots']['styles']
        assert 'linewidth' in config['plots']['styles']

        # Output
        assert 'files' in config['output']
        assert 'table' in config['output']

        assert 'experiment_index' in config['output']['files']
        assert 'latency_vs_throughput_plot' in config['output']['files']
        assert 'latency_vs_throughput_table' in config['output']['files']
        assert 'success_vs_load_plot' in config['output']['files']
        assert 'success_vs_throughput_plot' in config['output']['files']
        assert 'overhead_vs_throughput_plot' in config['output']['files']
        assert 'overhead_vs_throughput_table' in config['output']['files']

        assert 'float_format' in config['output']['table']
        assert 'na_rep' in config['output']['table']
        assert 'index' in config['output']['table']

        print("✓ All default config keys present")

    def test_saturation_config_defaults(self):
        """Verify saturation configuration has correct defaults."""
        config = get_default_config()

        # Check saturation section exists
        assert 'saturation' in config['plots']

        # Check default values
        sat_config = config['plots']['saturation']
        assert sat_config['enabled'] is False  # Disabled by default
        assert sat_config['threshold'] == 50.0
        assert sat_config['tolerance'] == 5.0

        print("✓ Saturation config has correct defaults")

    def test_saturation_config_override(self):
        """Verify saturation configuration can be overridden."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "sat_config.toml"

            # Create config with custom saturation settings
            config_content = """
[plots.saturation]
enabled = false
threshold = 60.0
tolerance = 10.0
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = load_config(str(config_path))

            # Verify custom values
            sat_config = config['plots']['saturation']
            assert sat_config['enabled'] is False
            assert sat_config['threshold'] == 60.0
            assert sat_config['tolerance'] == 10.0

            # Default values should still be present for other settings
            assert config['plots']['dpi'] == saturation_analysis.DEFAULT_DPI

            print("✓ Saturation config can be overridden")

    def test_stddev_config_defaults(self):
        """Verify stddev configuration has correct defaults."""
        config = get_default_config()

        # Check stddev section exists
        assert 'stddev' in config['plots']

        # Check default values
        stddev_config = config['plots']['stddev']
        assert stddev_config['enabled'] is True
        assert stddev_config['alpha'] == 0.2

        print("✓ Stddev config has correct defaults")

    def test_stddev_config_override(self):
        """Verify stddev configuration can be overridden."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "stddev_config.toml"

            # Create config with custom stddev settings
            config_content = """
[plots.stddev]
enabled = false
alpha = 0.3
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            config = load_config(str(config_path))

            # Verify custom values
            stddev_config = config['plots']['stddev']
            assert stddev_config['enabled'] is False
            assert stddev_config['alpha'] == 0.3

            print("✓ Stddev config can be overridden")

    def test_per_seed_statistics_computed(self):
        """Verify that per-seed statistics are computed correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data with multiple seeds
            seed_dirs = []
            for seed in [111, 222]:
                seed_dir = Path(tmpdir) / str(seed)
                seed_dir.mkdir()
                seed_dirs.append(str(seed_dir))

                n_txns = 60
                df = pd.DataFrame({
                    'txn_id': range(n_txns),
                    't_submit': [i * 60000 for i in range(n_txns)],
                    't_commit': [(i * 60000) + 11000 for i in range(n_txns)],
                    'commit_latency': [1000 + seed] * n_txns,  # Different per seed
                    'total_latency': [11000] * n_txns,
                    'n_retries': [0] * n_txns,
                    'status': ['committed'] * n_txns
                })
                df.to_parquet(seed_dir / "results.parquet")

            # Load and compute statistics
            exp_info = {
                'seeds': seed_dirs,
                'config': {
                    'transaction': {'runtime': {'mean': 10000}},
                    'simulation': {'duration_ms': 3600000}
                }
            }
            result_df = load_and_aggregate_results(exp_info)
            stats = compute_aggregate_statistics(result_df)

            # Check that stddev columns exist
            assert 'p50_commit_latency_std' in stats
            assert 'p95_commit_latency_std' in stats
            assert 'p99_commit_latency_std' in stats
            assert 'success_rate_std' in stats
            assert 'throughput_std' in stats

            # With 2 seeds having different latencies (1111 and 1222),
            # stddev should be non-zero
            assert stats['p50_commit_latency_std'] > 0

            print("✓ Per-seed statistics computed correctly")

    def test_cli_overrides_config_file(self):
        """Verify CLI arguments override config file values."""
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config file
            config_path = Path(tmpdir) / "test_config.toml"
            config_content = """
[paths]
input_dir = "config_experiments"
output_dir = "config_plots"
pattern = "config_*"

[analysis]
group_by = "num_tables"
min_seeds = 1
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            # Create experiment structure with complete config
            exp_dir = Path(tmpdir) / "cli_exp-abc123"
            exp_dir.mkdir()

            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write("""
[simulation]
duration_ms = 3600000

[experiment]
label = 'cli_exp'

[catalog]
num_tables = 1

[transaction]
runtime.mean = 10000
inter_arrival.scale = 100.0
retry = 10
""")

            seed_dir = exp_dir / "12345"
            seed_dir.mkdir()

            # Create results spanning 1 hour with enough data after warmup/cooldown
            # With 15-min warmup and 15-min cooldown, active window is 30 minutes (900-2700 seconds)
            # Generate 300 transactions (one every 12 seconds) to ensure enough data
            n_txns = 300
            df = pd.DataFrame({
                'txn_id': range(n_txns),
                't_submit': [i * 12000 for i in range(n_txns)],  # Every 12 seconds for 60 minutes
                't_runtime': [10000] * n_txns,
                't_commit': [(i * 12000) + 11000 for i in range(n_txns)],
                'commit_latency': [1000] * n_txns,
                'total_latency': [11000] * n_txns,
                'n_retries': [0] * n_txns,
                'n_tables_read': [1] * n_txns,
                'n_tables_written': [1] * n_txns,
                'status': ['committed'] * n_txns
            })
            df.to_parquet(seed_dir / "results.parquet")

            output_dir = Path(tmpdir) / "cli_output"

            # Run CLI with overrides
            result = subprocess.run([
                sys.executable, "-m", "endive.saturation_analysis",
                "--config", str(config_path),
                "-i", str(tmpdir),
                "-o", str(output_dir),
                "-p", "cli_exp-*"
            ], capture_output=True, text=True)

            # Check that CLI ran successfully
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that output was created in CLI-specified directory (not config directory)
            assert output_dir.exists()
            assert (output_dir / "experiment_index.csv").exists()

            # Load the index and verify it found the experiment
            index = pd.read_csv(output_dir / "experiment_index.csv")
            assert len(index) == 1
            assert 'cli_exp' in index['label'].values[0]

            print("✓ CLI overrides config file values")

    def test_cli_without_config_uses_defaults(self):
        """Verify CLI works without config file using defaults."""
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create experiment structure with complete config
            exp_dir = Path(tmpdir) / "default_exp-xyz789"
            exp_dir.mkdir()

            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write("""
[simulation]
duration_ms = 3600000

[experiment]
label = 'default_exp'

[catalog]
num_tables = 1

[transaction]
runtime.mean = 10000
inter_arrival.scale = 100.0
retry = 10
""")

            # Create 3 seeds to meet min_seeds default of 3
            for seed in ["12345", "23456", "34567"]:
                seed_dir = exp_dir / seed
                seed_dir.mkdir()

                # Create results spanning 1 hour with enough data after warmup/cooldown
                n_txns = 300
                df = pd.DataFrame({
                    'txn_id': range(n_txns),
                    't_submit': [i * 12000 for i in range(n_txns)],  # Every 12 seconds for 60 minutes
                    't_runtime': [10000] * n_txns,
                    't_commit': [(i * 12000) + 11000 for i in range(n_txns)],
                    'commit_latency': [1000] * n_txns,
                    'total_latency': [11000] * n_txns,
                    'n_retries': [0] * n_txns,
                    'n_tables_read': [1] * n_txns,
                    'n_tables_written': [1] * n_txns,
                    'status': ['committed'] * n_txns
                })
                df.to_parquet(seed_dir / "results.parquet")

            output_dir = Path(tmpdir) / "default_output"

            # Run CLI without config file
            result = subprocess.run([
                sys.executable, "-m", "endive.saturation_analysis",
                "-i", str(tmpdir),
                "-o", str(output_dir),
                "-p", "default_exp-*"
            ], capture_output=True, text=True)

            # Check that CLI ran successfully
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that output was created
            assert output_dir.exists()
            assert (output_dir / "experiment_index.csv").exists()

            # Load the index
            index = pd.read_csv(output_dir / "experiment_index.csv")
            assert len(index) == 1

            print("✓ CLI works without config file")

    def test_cli_group_by_override(self):
        """Verify CLI --group-by argument overrides config."""
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config without group_by (omit it to test override)
            config_path = Path(tmpdir) / "group_config.toml"
            config_content = """
[analysis]
# group_by not specified - will be overridden by CLI
min_seeds = 1
"""
            with open(config_path, 'w') as f:
                f.write(config_content)

            # Create two experiments with different num_tables
            for num_tables in [1, 5]:
                exp_dir = Path(tmpdir) / f"group_exp_{num_tables}-hash{num_tables}"
                exp_dir.mkdir()

                with open(exp_dir / "cfg.toml", 'w') as f:
                    f.write(f"""
[simulation]
duration_ms = 3600000

[experiment]
label = "group_exp_{num_tables}"

[catalog]
num_tables = {num_tables}

[transaction]
inter_arrival.scale = 100
runtime.mean = 10000
retry = 10
""")

                seed_dir = exp_dir / "12345"
                seed_dir.mkdir()

                # Create results spanning 1 hour with enough data after warmup/cooldown
                n_txns = 300
                df = pd.DataFrame({
                    'txn_id': range(n_txns),
                    't_submit': [i * 12000 for i in range(n_txns)],  # Every 12 seconds for 60 minutes
                    't_runtime': [10000] * n_txns,
                    't_commit': [(i * 12000) + 11000 for i in range(n_txns)],
                    'commit_latency': [1000] * n_txns,
                    'total_latency': [11000] * n_txns,
                    'n_retries': [0] * n_txns,
                    'n_tables_read': [num_tables] * n_txns,
                    'n_tables_written': [num_tables] * n_txns,
                    'status': ['committed'] * n_txns
                })
                df.to_parquet(seed_dir / "results.parquet")

            output_dir = Path(tmpdir) / "group_output"

            # Run CLI with --group-by override
            result = subprocess.run([
                sys.executable, "-m", "endive.saturation_analysis",
                "--config", str(config_path),
                "-i", str(tmpdir),
                "-o", str(output_dir),
                "-p", "group_exp_*",
                "--group-by", "num_tables"
            ], capture_output=True, text=True)

            # Check that CLI ran successfully
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            # Check that output was created
            assert output_dir.exists()

            # Load index and verify grouping parameter was extracted
            index = pd.read_csv(output_dir / "experiment_index.csv")
            assert len(index) == 2
            assert 'num_tables' in index.columns
            assert set(index['num_tables'].values) == {1, 5}

            print("✓ CLI --group-by overrides config")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
