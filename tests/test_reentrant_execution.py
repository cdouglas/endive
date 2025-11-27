"""Tests for reentrant experiment execution functionality.

Verifies that:
1. Completed experiments are correctly detected and skipped
2. Hash mismatches produce warnings but allow continuation
3. New seeds in existing experiments are allowed to run
4. Partial results (incomplete runs) are not considered complete
"""

import os
import tempfile
import pytest
import shutil
from pathlib import Path
import tomllib

from icecap.main import check_existing_experiment, compute_experiment_hash, prepare_experiment_output
from icecap.test_utils import create_test_config


class TestCheckExistingExperiment:
    """Test the check_existing_experiment function."""

    def test_no_experiment_label_checks_simple_path(self):
        """When no experiment.label, check if output file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = create_test_config(
                output_path="results.parquet",
                seed=42
            )

            try:
                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Without results file, should not skip
                skip, seed, output = check_existing_experiment(config, config_path)
                assert skip is False
                assert seed is None
                assert output is None

                # Create results file
                Path("results.parquet").touch()

                try:
                    # With results file, should skip
                    skip, seed, output = check_existing_experiment(config, config_path)
                    assert skip is True
                    assert seed is None
                    assert output == "results.parquet"

                    print(f"✓ Simple path existence check works")
                finally:
                    Path("results.parquet").unlink()

            finally:
                os.unlink(config_path)

    def test_with_label_checks_experiment_directory(self):
        """With experiment.label, check experiments/$label-$hash/$seed/."""
        with tempfile.TemporaryDirectory() as tmpdir:
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

                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Without experiment directory, should not skip
                skip, seed, output = check_existing_experiment(config, config_path)
                assert skip is False

                # Create experiment structure
                exp_hash = compute_experiment_hash(config)
                exp_dir = Path("experiments") / f"test_exp-{exp_hash}"
                seed_dir = exp_dir / "42"
                results_path = seed_dir / "results.parquet"

                seed_dir.mkdir(parents=True)
                results_path.touch()

                # Write cfg.toml
                (exp_dir / "cfg.toml").write_text(Path(config_path).read_text())

                # Now should skip
                skip, seed, output = check_existing_experiment(config, config_path)
                assert skip is True
                assert seed == 42
                assert "test_exp" in output
                assert "42" in output

                print(f"✓ Experiment directory check works")

            finally:
                os.chdir(original_cwd)
                os.unlink(config_path)

    def test_incomplete_results_not_skipped(self):
        """Partial results (.running.parquet) should not cause skip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config_path = create_test_config(
                    output_path="results.parquet",
                    seed=42
                )

                with open(config_path, 'r') as f:
                    content = f.read()
                content = content.replace(
                    '[catalog]',
                    '[experiment]\nlabel = "test_partial"\n\n[catalog]'
                )
                with open(config_path, 'w') as f:
                    f.write(content)

                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Create experiment with only .running.parquet (incomplete)
                exp_hash = compute_experiment_hash(config)
                exp_dir = Path("experiments") / f"test_partial-{exp_hash}"
                seed_dir = exp_dir / "42"
                running_path = seed_dir / ".running.parquet"

                seed_dir.mkdir(parents=True)
                running_path.touch()  # Incomplete run

                # Should NOT skip (only .running.parquet exists)
                skip, seed, output = check_existing_experiment(config, config_path)
                assert skip is False, "Should not skip incomplete results"

                print(f"✓ Incomplete results not skipped")

            finally:
                os.chdir(original_cwd)
                os.unlink(config_path)

    def test_new_seed_in_existing_experiment_allowed(self):
        """New seed in existing experiment should be allowed to run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # Run with seed 42
                config_path1 = create_test_config(output_path="results.parquet", seed=42)
                with open(config_path1, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "multi_seed"\n\n[catalog]')
                with open(config_path1, 'w') as f:
                    f.write(content)

                with open(config_path1, 'rb') as f:
                    config1 = tomllib.load(f)

                # Create seed 42 results
                exp_hash = compute_experiment_hash(config1)
                exp_dir = Path("experiments") / f"multi_seed-{exp_hash}"
                seed42_dir = exp_dir / "42"
                seed42_dir.mkdir(parents=True)
                (seed42_dir / "results.parquet").touch()
                (exp_dir / "cfg.toml").write_text(Path(config_path1).read_text())

                # Check seed 42 - should skip
                skip, seed, output = check_existing_experiment(config1, config_path1)
                assert skip is True
                assert seed == 42

                # Now try seed 99 - should NOT skip
                config_path2 = create_test_config(output_path="results.parquet", seed=99)
                with open(config_path2, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "multi_seed"\n\n[catalog]')
                with open(config_path2, 'w') as f:
                    f.write(content)

                with open(config_path2, 'rb') as f:
                    config2 = tomllib.load(f)

                skip, seed, output = check_existing_experiment(config2, config_path2)
                assert skip is False, "New seed should be allowed to run"

                print(f"✓ New seed in existing experiment allowed")

                os.unlink(config_path1)
                os.unlink(config_path2)

            finally:
                os.chdir(original_cwd)

    def test_hash_mismatch_warning(self, caplog):
        """Hash mismatch should produce warning but allow continuation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                config_path = create_test_config(
                    output_path="results.parquet",
                    seed=42,
                    num_tables=10
                )

                with open(config_path, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "hash_test"\n\n[catalog]')
                with open(config_path, 'w') as f:
                    f.write(content)

                with open(config_path, 'rb') as f:
                    config = tomllib.load(f)

                # Create experiment directory with seed 42
                exp_hash = compute_experiment_hash(config)
                exp_dir = Path("experiments") / f"hash_test-{exp_hash}"
                seed_dir = exp_dir / "42"
                seed_dir.mkdir(parents=True)
                (seed_dir / "results.parquet").touch()

                # Write cfg.toml with DIFFERENT config (different hash)
                different_config_path = create_test_config(
                    output_path="results.parquet",
                    seed=42,
                    num_tables=20  # Different!
                )
                with open(different_config_path, 'r') as f:
                    different_content = f.read()
                different_content = different_content.replace('[catalog]', '[experiment]\nlabel = "hash_test"\n\n[catalog]')
                (exp_dir / "cfg.toml").write_text(different_content)

                # Check experiment - should skip but log warning
                with caplog.at_level("WARNING"):
                    skip, seed, output = check_existing_experiment(config, config_path)

                # Should still skip (results exist for this seed)
                assert skip is True

                # Should have warning about hash mismatch
                warnings = [record for record in caplog.records if record.levelname == "WARNING"]
                assert len(warnings) > 0, "Should have logged hash mismatch warning"
                assert "Hash mismatch" in warnings[0].message or "mismatch" in warnings[0].message.lower()

                print(f"✓ Hash mismatch produces warning")

                os.unlink(config_path)
                os.unlink(different_config_path)

            finally:
                os.chdir(original_cwd)


class TestExperimentPrepareOutput:
    """Test prepare_experiment_output handles reentrant execution."""

    def test_second_seed_uses_existing_config(self):
        """Second seed should not overwrite existing cfg.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                # First run with seed 42
                config1 = create_test_config(output_path="results.parquet", seed=42)
                with open(config1, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "reuse_test"\n\n[catalog]')
                with open(config1, 'w') as f:
                    f.write(content)

                with open(config1, 'rb') as f:
                    cfg1 = tomllib.load(f)

                output1 = prepare_experiment_output(cfg1, config1, actual_seed=42)
                exp_dir = Path(output1).parent.parent
                cfg_path = exp_dir / "cfg.toml"

                # Record original modification time
                original_mtime = cfg_path.stat().st_mtime
                original_content = cfg_path.read_text()

                # Second run with seed 99
                config2 = create_test_config(output_path="results.parquet", seed=99)
                with open(config2, 'r') as f:
                    content = f.read()
                content = content.replace('[catalog]', '[experiment]\nlabel = "reuse_test"\n\n[catalog]')
                with open(config2, 'w') as f:
                    f.write(content)

                with open(config2, 'rb') as f:
                    cfg2 = tomllib.load(f)

                output2 = prepare_experiment_output(cfg2, config2, actual_seed=99)

                # cfg.toml should not be modified
                new_mtime = cfg_path.stat().st_mtime
                new_content = cfg_path.read_text()

                assert original_mtime == new_mtime, "cfg.toml should not be overwritten"
                assert original_content == new_content, "cfg.toml content should not change"

                print(f"✓ Config not overwritten on second seed")

                os.unlink(config1)
                os.unlink(config2)

            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
