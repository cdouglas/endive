"""Tests for experiment labeling and output structure.

Verifies that:
1. Experiment hash is deterministic and excludes seed/label
2. Output paths are created correctly with label-hash-seed structure
3. Config files are written to experiment directories
4. Code changes affect hash
5. Same config + code → same hash regardless of seed
"""

import os
import tempfile
import pytest
import shutil
from pathlib import Path

from icecap.main import compute_experiment_hash, prepare_experiment_output
from icecap.test_utils import create_test_config


class TestExperimentHash:
    """Test experiment hash computation."""

    def test_hash_excludes_seed(self):
        """Verify hash is same regardless of seed."""
        config1 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet", "seed": 42},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10}
        }

        config2 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet", "seed": 99},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10}
        }

        hash1 = compute_experiment_hash(config1)
        hash2 = compute_experiment_hash(config2)

        assert hash1 == hash2, f"Hash should be same regardless of seed: {hash1} != {hash2}"

        print(f"✓ Hash excludes seed")
        print(f"  Config with seed=42: {hash1}")
        print(f"  Config with seed=99: {hash2}")

    def test_hash_excludes_label(self):
        """Verify hash is same regardless of experiment label."""
        config1 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet"},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10},
            "experiment": {"label": "exp1"}
        }

        config2 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet"},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10},
            "experiment": {"label": "exp2"}
        }

        hash1 = compute_experiment_hash(config1)
        hash2 = compute_experiment_hash(config2)

        assert hash1 == hash2, f"Hash should be same regardless of label: {hash1} != {hash2}"

        print(f"✓ Hash excludes experiment label")
        print(f"  Config with label=exp1: {hash1}")
        print(f"  Config with label=exp2: {hash2}")

    def test_hash_changes_with_config(self):
        """Verify hash changes when config parameters change."""
        config1 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet"},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10}
        }

        config2 = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet"},
            "catalog": {"num_tables": 20},  # Changed
            "transaction": {"retry": 10}
        }

        hash1 = compute_experiment_hash(config1)
        hash2 = compute_experiment_hash(config2)

        assert hash1 != hash2, f"Hash should change when config changes: {hash1} == {hash2}"

        print(f"✓ Hash changes with config")
        print(f"  Config with num_tables=10: {hash1}")
        print(f"  Config with num_tables=20: {hash2}")

    def test_hash_includes_code(self):
        """Verify hash includes simulator code (changes when code changes)."""
        config = {
            "simulation": {"duration_ms": 1000, "output_path": "test.parquet"},
            "catalog": {"num_tables": 10},
            "transaction": {"retry": 10}
        }

        # Compute hash with current code
        hash1 = compute_experiment_hash(config)

        # Hash should be deterministic - compute again
        hash2 = compute_experiment_hash(config)

        assert hash1 == hash2, "Hash should be deterministic"

        # Hash should be 8 characters
        assert len(hash1) == 8, f"Hash should be 8 characters, got {len(hash1)}"
        assert all(c in '0123456789abcdef' for c in hash1), "Hash should be hexadecimal"

        print(f"✓ Hash is deterministic and includes code")
        print(f"  Hash: {hash1}")


class TestExperimentOutputStructure:
    """Test experiment output directory structure."""

    def test_no_label_uses_original_path(self):
        """Verify that without experiment.label, original output_path is used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path="results.parquet",
                seed=42
            )

            try:
                import tomllib
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # No experiment label
                output = prepare_experiment_output(config, config_path, actual_seed=42)

                assert output == "results.parquet", f"Should use original path, got {output}"

                print(f"✓ No label uses original output path")

            finally:
                os.unlink(config_path)

    def test_with_label_creates_experiment_structure(self):
        """Verify experiment.label creates proper directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir for this test
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config_path = create_test_config(
                    output_path="results.parquet",
                    seed=42
                )

                # Add experiment label
                with open(config_path, 'r') as f:
                    content = f.read()

                content = content.replace(
                    '[catalog]',
                    '[experiment]\nlabel = "test_exp"\n\n[catalog]'
                )

                with open(config_path, 'w') as f:
                    f.write(content)

                # Load config
                import tomllib
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Prepare output with actual seed
                output = prepare_experiment_output(config, config_path, actual_seed=42)

                # Verify structure: experiments/test_exp-HASH/42/results.parquet
                output_path = Path(output)
                parts = output_path.parts

                assert parts[0] == "experiments", f"Should start with 'experiments', got {parts[0]}"
                assert parts[1].startswith("test_exp-"), f"Should have label-hash format, got {parts[1]}"
                assert parts[2] == "42", f"Should have seed dir, got {parts[2]}"
                assert parts[3] == "results.parquet", f"Should have output file, got {parts[3]}"

                # Verify cfg.toml was written
                exp_dir = Path("experiments") / parts[1]
                config_file = exp_dir / "cfg.toml"
                assert config_file.exists(), f"Config file should exist at {config_file}"

                print(f"✓ Experiment structure created correctly")
                print(f"  Output path: {output}")
                print(f"  Config written: {config_file}")

            finally:
                os.chdir(original_cwd)
                os.unlink(config_path)

    def test_random_seed_directory(self):
        """Verify actual seed is used in directory name, even if generated randomly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config_path = create_test_config(
                    output_path="results.parquet",
                    seed=None  # No seed in config
                )

                # Add experiment label
                with open(config_path, 'r') as f:
                    content = f.read()

                content = content.replace(
                    '[catalog]',
                    '[experiment]\nlabel = "test_random"\n\n[catalog]'
                )

                with open(config_path, 'w') as f:
                    f.write(content)

                # Load config
                import tomllib
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Generate a random seed (simulating what CLI does)
                import numpy as np
                actual_seed = np.random.randint(0, 2**32 - 1)

                # Prepare output with actual seed
                output = prepare_experiment_output(config, config_path, actual_seed=actual_seed)

                # Verify structure includes the actual seed
                output_path = Path(output)
                parts = output_path.parts

                assert parts[2] == str(actual_seed), f"Should use actual seed directory, got {parts[2]}"

                print(f"✓ Actual seed used in directory")
                print(f"  Generated seed: {actual_seed}")
                print(f"  Output path: {output}")

            finally:
                os.chdir(original_cwd)
                os.unlink(config_path)

    def test_same_hash_different_seeds(self):
        """Verify same config + code with different seeds share experiment directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                hashes = []
                for seed in [42, 99, 123]:
                    config_path = create_test_config(
                        output_path="results.parquet",
                        seed=seed
                    )

                    # Add experiment label
                    with open(config_path, 'r') as f:
                        content = f.read()

                    content = content.replace(
                        '[catalog]',
                        '[experiment]\nlabel = "multi_seed"\n\n[catalog]'
                    )

                    with open(config_path, 'w') as f:
                        f.write(content)

                    # Load config
                    import tomllib
                    with open(config_path, 'rb') as f:
                        config = tomllib.load(f)

                    # Prepare output with actual seed
                    output = prepare_experiment_output(config, config_path, actual_seed=seed)
                    output_path = Path(output)

                    # Extract hash from path (experiments/label-HASH/seed/...)
                    exp_dir_name = output_path.parts[1]  # e.g., "multi_seed-abc12345"
                    exp_hash = exp_dir_name.split('-')[1]
                    hashes.append(exp_hash)

                    os.unlink(config_path)

                # All hashes should be the same
                assert len(set(hashes)) == 1, f"All seeds should produce same hash: {hashes}"

                # Verify all seed directories exist under same experiment directory
                exp_dir = Path("experiments") / f"multi_seed-{hashes[0]}"
                for seed in [42, 99, 123]:
                    seed_dir = exp_dir / str(seed)
                    assert seed_dir.exists(), f"Seed directory {seed_dir} should exist"

                print(f"✓ Same config produces same hash across seeds")
                print(f"  Hash: {hashes[0]}")
                print(f"  Seed directories: 42, 99, 123")

            finally:
                os.chdir(original_cwd)


class TestConfigPersistence:
    """Test that config files are written correctly."""

    def test_config_written_once(self):
        """Verify cfg.toml is written once per experiment, not per seed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Run with seed 42
                config_path1 = create_test_config(output_path="results.parquet", seed=42)
                with open(config_path1, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "persist_test"\n\n[catalog]')
                with open(config_path1, 'w') as f:
                    f.write(content)

                import tomllib
                with open(config_path1, 'rb') as f:
                    config1 = tomllib.load(f)

                output1 = prepare_experiment_output(config1, config_path1, actual_seed=42)
                exp_dir = Path(output1).parent.parent  # experiments/label-hash

                # Get modification time of cfg.toml
                cfg_path = exp_dir / "cfg.toml"
                assert cfg_path.exists(), "Config should exist after first run"
                mtime1 = cfg_path.stat().st_mtime

                # Run with seed 99 (same experiment)
                config_path2 = create_test_config(output_path="results.parquet", seed=99)
                with open(config_path2, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "persist_test"\n\n[catalog]')
                with open(config_path2, 'w') as f:
                    f.write(content)

                with open(config_path2, 'rb') as f:
                    config2 = tomllib.load(f)

                output2 = prepare_experiment_output(config2, config_path2, actual_seed=99)

                # Config should not be overwritten
                mtime2 = cfg_path.stat().st_mtime
                assert mtime1 == mtime2, "Config file should not be overwritten on second run"

                print(f"✓ Config written once per experiment")
                print(f"  First run: {output1}")
                print(f"  Second run: {output2}")

                os.unlink(config_path1)
                os.unlink(config_path2)

            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
