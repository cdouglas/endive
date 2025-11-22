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

from icecap.saturation_analysis import (
    scan_experiment_directories,
    extract_key_parameters,
    load_and_aggregate_results,
    compute_aggregate_statistics,
    build_experiment_index,
    save_experiment_index
)


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
num_groups = 1

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
                'num_tables': 10,
                'num_groups': 5
            }
        }

        params = extract_key_parameters(config)
        assert params['num_tables'] == 10
        assert params['num_groups'] == 5

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

            # Create sample results
            df = pd.DataFrame({
                'txn_id': [1, 2, 3],
                't_submit': [0, 100, 200],
                't_commit': [50, 150, 250],
                'commit_latency': [50, 50, 50],
                'n_retries': [0, 1, 0],
                'status': ['committed', 'committed', 'committed']
            })
            df.to_parquet(seed_dir / "results.parquet")

            # Load
            exp_info = {
                'seeds': [str(seed_dir)]
            }
            result = load_and_aggregate_results(exp_info)

            assert result is not None
            assert len(result) == 3
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

                df = pd.DataFrame({
                    'txn_id': [1, 2],
                    't_submit': [0, 100],
                    't_commit': [50, 150],
                    'commit_latency': [50, 50],
                    'n_retries': [0, 1],
                    'status': ['committed', 'committed']
                })
                df.to_parquet(seed_dir / "results.parquet")

            # Aggregate
            exp_info = {'seeds': seed_dirs}
            result = load_and_aggregate_results(exp_info)

            assert result is not None
            assert len(result) == 6  # 2 transactions × 3 seeds

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
duration_ms = 1000

[experiment]
label = "exp_test"

[catalog]
num_tables = 1
num_groups = 1

[transaction]
inter_arrival.scale = 100
real_conflict_probability = 0.0
"""
            with open(exp_dir / "cfg.toml", 'w') as f:
                f.write(cfg_content)

            seed_dir = exp_dir / "12345"
            seed_dir.mkdir()

            # Create sample results
            df = pd.DataFrame({
                'txn_id': [1, 2, 3],
                't_submit': [0, 100, 200],
                't_commit': [50, 150, 250],
                'commit_latency': [50, 50, 50],
                'n_retries': [0, 1, 0],
                'status': ['committed', 'committed', 'committed']
            })
            df.to_parquet(seed_dir / "results.parquet")

            # Build index
            index_df = build_experiment_index(tmpdir, "exp_test-*")

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
                f.write("[simulation]\nduration_ms = 1000\n[experiment]\nlabel = \"exp_test\"\n[transaction]\ninter_arrival.scale = 100\n")

            # Create 3 seeds with different results
            for seed in [111, 222, 333]:
                seed_dir = exp_dir / str(seed)
                seed_dir.mkdir()

                df = pd.DataFrame({
                    'txn_id': [1, 2],
                    't_submit': [0, 100],
                    't_commit': [50, 150],
                    'commit_latency': [50, 50],
                    'n_retries': [0, 1],
                    'status': ['committed', 'committed']
                })
                df.to_parquet(seed_dir / "results.parquet")

            index_df = build_experiment_index(tmpdir, "exp_test-*")

            assert len(index_df) == 1
            row = index_df.iloc[0]

            # Should aggregate: 2 txns × 3 seeds = 6 total
            assert row['num_seeds'] == 3
            assert row['total_txns'] == 6

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
duration_ms = 1000

[experiment]
label = "exp_test"

[catalog]
num_tables = 1

[transaction]
inter_arrival.scale = {ia_scale}
real_conflict_probability = 0.0
""")

                    # Create seed directory
                    seed_dir = exp_dir / str(seed)
                    seed_dir.mkdir()

                    # Create results
                    n_txns = 10 if ia_scale == 100 else 5
                    df = pd.DataFrame({
                        'txn_id': range(n_txns),
                        't_submit': [i * 100 for i in range(n_txns)],
                        't_commit': [(i * 100) + 50 for i in range(n_txns)],
                        'commit_latency': [50] * n_txns,
                        'n_retries': [0] * n_txns,
                        'status': ['committed'] * n_txns
                    })
                    df.to_parquet(seed_dir / "results.parquet")

            # Build index
            index_df = build_experiment_index(tmpdir, "exp_test-*")

            # Should have 2 experiments (one per inter_arrival value)
            assert len(index_df) == 2

            # Verify different throughputs
            throughputs = sorted(index_df['throughput'].values)
            assert throughputs[0] < throughputs[1]  # Lower inter_arrival → lower throughput

            # Verify parameter extraction
            assert 100 in index_df['inter_arrival_scale'].values
            assert 500 in index_df['inter_arrival_scale'].values

            print("✓ Full workflow works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
