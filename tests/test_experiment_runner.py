"""
Tests for the experiment runner script (run_all_experiments.py).

These tests validate:
1. Config variant creation with parameter substitution
2. Dotted key handling (e.g., inter_arrival.scale)
3. Seed injection
4. Duration override
5. Experiment generation for all groups
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_all_experiments import (
    create_config_variant,
    generate_seed,
    ExperimentRun,
    EXPERIMENT_GROUPS,
    LOAD_SWEEP,
    CONFIG_DIR,
)


class TestCreateConfigVariant:
    """Test config variant creation with parameter substitution."""

    def test_dotted_key_substitution(self, tmp_path):
        """Dotted keys like inter_arrival.scale should be substituted correctly."""
        # Create a base config
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
""")

        # Create variant with different inter_arrival.scale
        variant_path = create_config_variant(
            base_config,
            params={"inter_arrival.scale": 500.0},
            seed=42
        )

        try:
            content = Path(variant_path).read_text()

            # Should have the new value
            assert "inter_arrival.scale = 500.0" in content
            # Should NOT have the old value
            assert "inter_arrival.scale = 100.0" not in content
            # Seed should be injected
            assert "seed = 42" in content
        finally:
            os.unlink(variant_path)

    def test_simple_key_substitution(self, tmp_path):
        """Simple keys without dots should be substituted correctly."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000

[catalog]
num_tables = 1

[transaction]
real_conflict_probability = 0.0
""")

        variant_path = create_config_variant(
            base_config,
            params={"num_tables": 10, "real_conflict_probability": 0.5},
            seed=123
        )

        try:
            content = Path(variant_path).read_text()

            assert "num_tables = 10" in content
            assert "real_conflict_probability = 0.5" in content
            assert "seed = 123" in content
        finally:
            os.unlink(variant_path)

    def test_duration_override(self, tmp_path):
        """Duration should be overridable."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000
""")

        variant_path = create_config_variant(
            base_config,
            params={},
            seed=1,
            duration_ms=60000
        )

        try:
            content = Path(variant_path).read_text()
            assert "duration_ms = 60000" in content
            assert "duration_ms = 3600000" not in content
        finally:
            os.unlink(variant_path)

    def test_seed_injection_after_simulation_section(self, tmp_path):
        """Seed should be injected right after [simulation] section."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000
output_path = "results.parquet"

[catalog]
num_tables = 1
""")

        variant_path = create_config_variant(
            base_config,
            params={},
            seed=999
        )

        try:
            content = Path(variant_path).read_text()
            # Seed should appear after [simulation] but before duration_ms
            sim_pos = content.find("[simulation]")
            seed_pos = content.find("seed = 999")
            duration_pos = content.find("duration_ms")

            assert sim_pos < seed_pos < duration_pos
        finally:
            os.unlink(variant_path)

    def test_multiple_dotted_keys(self, tmp_path):
        """Multiple dotted keys should all be substituted."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[transaction]
runtime.min = 30000
runtime.mean = 180000
runtime.sigma = 1.5
inter_arrival.scale = 100.0
conflicting_manifests.mean = 3.0
""")

        variant_path = create_config_variant(
            base_config,
            params={
                "inter_arrival.scale": 200.0,
                "conflicting_manifests.mean": 5.0
            },
            seed=1
        )

        try:
            content = Path(variant_path).read_text()

            # Changed values
            assert "inter_arrival.scale = 200.0" in content
            assert "conflicting_manifests.mean = 5.0" in content

            # Unchanged values
            assert "runtime.min = 30000" in content
            assert "runtime.mean = 180000" in content
        finally:
            os.unlink(variant_path)

    def test_preserves_comments_and_structure(self, tmp_path):
        """Config variant should preserve comments and overall structure."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""# Baseline: S3 with no optimizations
#
# Sweep inter_arrival.scale: [10, 20, 50, 100]

[simulation]
duration_ms = 3600000  # 1 hour

[storage]
provider = "s3"
""")

        variant_path = create_config_variant(
            base_config,
            params={},
            seed=42
        )

        try:
            content = Path(variant_path).read_text()

            # Comments preserved
            assert "# Baseline: S3 with no optimizations" in content
            assert "# 1 hour" in content

            # Structure preserved
            assert '[simulation]' in content
            assert '[storage]' in content
            assert 'provider = "s3"' in content
        finally:
            os.unlink(variant_path)


class TestGenerateSeed:
    """Test deterministic seed generation."""

    def test_same_inputs_same_seed(self):
        """Same inputs should always produce the same seed."""
        seed1 = generate_seed("nonce123", "exp1", {"load": 100}, 1)
        seed2 = generate_seed("nonce123", "exp1", {"load": 100}, 1)
        assert seed1 == seed2

    def test_different_nonce_different_seed(self):
        """Different nonces should produce different seeds."""
        seed1 = generate_seed("nonce123", "exp1", {"load": 100}, 1)
        seed2 = generate_seed("nonce456", "exp1", {"load": 100}, 1)
        assert seed1 != seed2

    def test_different_label_different_seed(self):
        """Different labels should produce different seeds."""
        seed1 = generate_seed("nonce", "exp1", {"load": 100}, 1)
        seed2 = generate_seed("nonce", "exp2", {"load": 100}, 1)
        assert seed1 != seed2

    def test_different_params_different_seed(self):
        """Different params should produce different seeds."""
        seed1 = generate_seed("nonce", "exp1", {"load": 100}, 1)
        seed2 = generate_seed("nonce", "exp1", {"load": 200}, 1)
        assert seed1 != seed2

    def test_different_seed_num_different_seed(self):
        """Different seed numbers should produce different seeds."""
        seed1 = generate_seed("nonce", "exp1", {"load": 100}, 1)
        seed2 = generate_seed("nonce", "exp1", {"load": 100}, 2)
        assert seed1 != seed2

    def test_seed_is_integer(self):
        """Generated seed should be an integer."""
        seed = generate_seed("nonce", "label", {}, 1)
        assert isinstance(seed, int)
        assert seed > 0


class TestExperimentRun:
    """Test ExperimentRun dataclass."""

    def test_run_id_format(self):
        """Run ID should include label, params, and seed."""
        run = ExperimentRun(
            config_path="test.toml",
            label="baseline_s3",
            seed=42,
            params={"inter_arrival.scale": 100.0}
        )

        assert run.run_id == "baseline_s3_inter_arrival.scale=100.0_s42"

    def test_param_str_sorted(self):
        """Param string should be sorted for consistency."""
        run = ExperimentRun(
            config_path="test.toml",
            label="test",
            seed=1,
            params={"z_param": 1, "a_param": 2, "m_param": 3}
        )

        assert run.param_str == "a_param=2_m_param=3_z_param=1"


class TestExperimentGroups:
    """Test experiment group definitions."""

    def test_all_groups_have_configs(self):
        """All experiment groups should have at least one config."""
        for group, configs in EXPERIMENT_GROUPS.items():
            assert len(configs) > 0, f"Group {group} has no configs"

    def test_baseline_group_configs_exist(self):
        """Baseline group configs should exist in experiment_configs."""
        for config in EXPERIMENT_GROUPS.get("baseline", []):
            config_path = CONFIG_DIR / config
            assert config_path.exists(), f"Missing config: {config_path}"

    def test_heatmap_group_configs_exist(self):
        """Heatmap group configs should exist."""
        for config in EXPERIMENT_GROUPS.get("heatmap", []):
            config_path = CONFIG_DIR / config
            assert config_path.exists(), f"Missing config: {config_path}"

    def test_catalog_group_configs_exist(self):
        """Catalog group configs should exist."""
        for config in EXPERIMENT_GROUPS.get("catalog", []):
            config_path = CONFIG_DIR / config
            assert config_path.exists(), f"Missing config: {config_path}"

    def test_all_configs_have_experiment_label(self):
        """All configs should have an [experiment] label."""
        import tomli

        for group, configs in EXPERIMENT_GROUPS.items():
            for config in configs:
                config_path = CONFIG_DIR / config
                if config_path.exists():
                    with open(config_path, "rb") as f:
                        data = tomli.load(f)
                    assert "experiment" in data, f"{config} missing [experiment] section"
                    assert "label" in data["experiment"], f"{config} missing experiment.label"


class TestLoadSweepValues:
    """Test load sweep parameter values."""

    def test_load_sweep_has_expected_range(self):
        """Load sweep should cover expected range of values."""
        assert min(LOAD_SWEEP) <= 20  # Should include low load
        assert max(LOAD_SWEEP) >= 2000  # Should include high load

    def test_load_sweep_is_sorted(self):
        """Load sweep values should be in ascending order."""
        assert LOAD_SWEEP == sorted(LOAD_SWEEP)

    def test_quick_loads_subset(self):
        """Quick loads should be a subset of full load sweep."""
        from run_all_experiments import QUICK_LOADS
        for load in QUICK_LOADS:
            assert load in LOAD_SWEEP


class TestConfigValidation:
    """Integration tests for config file validation."""

    @pytest.fixture
    def real_baseline_config(self):
        """Get path to real baseline S3 config."""
        return CONFIG_DIR / "baseline_s3.toml"

    def test_baseline_config_can_be_parsed(self, real_baseline_config):
        """Real baseline config should be parseable."""
        import tomli

        if real_baseline_config.exists():
            with open(real_baseline_config, "rb") as f:
                data = tomli.load(f)

            assert "simulation" in data
            assert "transaction" in data
            assert "storage" in data

    def test_baseline_config_has_sweep_parameter(self, real_baseline_config):
        """Baseline config should have inter_arrival.scale for sweeping."""
        if real_baseline_config.exists():
            content = real_baseline_config.read_text()
            assert "inter_arrival.scale" in content

    def test_variant_creation_with_real_config(self, real_baseline_config):
        """Creating variant from real config should work."""
        if not real_baseline_config.exists():
            pytest.skip("Baseline config not found")

        variant_path = create_config_variant(
            real_baseline_config,
            params={"inter_arrival.scale": 999.0},
            seed=12345,
            duration_ms=60000
        )

        try:
            import tomli
            with open(variant_path, "rb") as f:
                data = tomli.load(f)

            # Check seed was injected
            assert data["simulation"]["seed"] == 12345

            # Check duration was overridden
            assert data["simulation"]["duration_ms"] == 60000

            # Check inter_arrival.scale was substituted
            assert data["transaction"]["inter_arrival"]["scale"] == 999.0
        finally:
            os.unlink(variant_path)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_params(self, tmp_path):
        """Empty params should still create valid variant with seed."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 1000
""")

        variant_path = create_config_variant(
            base_config,
            params={},
            seed=1
        )

        try:
            content = Path(variant_path).read_text()
            assert "seed = 1" in content
        finally:
            os.unlink(variant_path)

    def test_special_characters_in_value(self, tmp_path):
        """Values should be substituted correctly even with special chars."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[transaction]
inter_arrival.scale = 100.0
""")

        # Float with many decimals
        variant_path = create_config_variant(
            base_config,
            params={"inter_arrival.scale": 123.456789},
            seed=1
        )

        try:
            content = Path(variant_path).read_text()
            assert "inter_arrival.scale = 123.456789" in content
        finally:
            os.unlink(variant_path)

    def test_integer_value_substitution(self, tmp_path):
        """Integer values should be substituted correctly."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[catalog]
num_tables = 1
""")

        variant_path = create_config_variant(
            base_config,
            params={"num_tables": 50},
            seed=1
        )

        try:
            content = Path(variant_path).read_text()
            assert "num_tables = 50" in content
        finally:
            os.unlink(variant_path)


class TestNestedTomlKeySubstitution:
    """Test substitution of nested TOML keys like partition.num_partitions.

    These are keys where the param name includes a section prefix (e.g., 'partition.num_partitions')
    but the TOML has the key inside a section block:

        [partition]
        num_partitions = 10

    NOT as a flat dotted key like inter_arrival.scale.
    """

    def test_nested_section_key_substitution(self, tmp_path):
        """Keys with section prefix should substitute within the section block."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000

[partition]
enabled = true
num_partitions = 10

[transaction]
retry = 5
""")

        variant_path = create_config_variant(
            base_config,
            params={"partition.num_partitions": 100},
            seed=42
        )

        try:
            content = Path(variant_path).read_text()
            # Should have new value
            assert "num_partitions = 100" in content
            # Should NOT have old value (check for exact match with newline)
            assert "num_partitions = 10\n" not in content
            # Other sections should be unchanged
            assert "enabled = true" in content
            assert "retry = 5" in content
        finally:
            os.unlink(variant_path)

    def test_nested_key_with_flat_key_in_same_config(self, tmp_path):
        """Both nested section keys and flat dotted keys should work together."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 3600000

[partition]
enabled = true
num_partitions = 10

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
""")

        variant_path = create_config_variant(
            base_config,
            params={
                "partition.num_partitions": 50,  # Nested section key
                "inter_arrival.scale": 200.0,    # Flat dotted key
            },
            seed=123
        )

        try:
            content = Path(variant_path).read_text()
            # Nested key substituted
            assert "num_partitions = 50" in content
            assert "num_partitions = 10" not in content
            # Flat key substituted
            assert "inter_arrival.scale = 200.0" in content
            assert "inter_arrival.scale = 100.0" not in content
        finally:
            os.unlink(variant_path)

    def test_multiple_nested_section_keys(self, tmp_path):
        """Multiple keys in same section should all be substitutable."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[partition]
enabled = true
num_partitions = 10

[partition.selection]
distribution = "uniform"

[partition.partitions_per_txn]
mean = 3.0
max = 10
""")

        variant_path = create_config_variant(
            base_config,
            params={
                "partition.num_partitions": 100,
            },
            seed=1
        )

        try:
            content = Path(variant_path).read_text()
            assert "num_partitions = 100" in content
            # Other partition settings unchanged
            assert "enabled = true" in content
        finally:
            os.unlink(variant_path)

    def test_nested_key_preserves_section_structure(self, tmp_path):
        """Substitution should preserve TOML section structure."""
        base_config = tmp_path / "test.toml"
        base_config.write_text("""
[simulation]
duration_ms = 1000

[partition]
enabled = true
num_partitions = 5
# Comment about partitions

[storage]
provider = "s3"
""")

        variant_path = create_config_variant(
            base_config,
            params={"partition.num_partitions": 25},
            seed=1
        )

        try:
            content = Path(variant_path).read_text()
            # Should have new value
            assert "num_partitions = 25" in content
            # Comment should be preserved
            assert "# Comment about partitions" in content
            # Sections should be in order
            sim_pos = content.find("[simulation]")
            part_pos = content.find("[partition]")
            stor_pos = content.find("[storage]")
            assert sim_pos < part_pos < stor_pos
        finally:
            os.unlink(variant_path)

    def test_real_partition_scaling_config(self):
        """Test with real partition scaling config if it exists."""
        config_path = CONFIG_DIR / "instant_partition_scaling.toml"
        if not config_path.exists():
            pytest.skip("Partition scaling config not found")

        variant_path = create_config_variant(
            config_path,
            params={
                "partition.num_partitions": 42,
                "inter_arrival.scale": 999.0,
            },
            seed=12345
        )

        try:
            import tomli
            with open(variant_path, "rb") as f:
                data = tomli.load(f)

            # Check partition.num_partitions was substituted
            assert data["partition"]["num_partitions"] == 42

            # Check inter_arrival.scale was substituted
            assert data["transaction"]["inter_arrival"]["scale"] == 999.0

            # Check seed was injected
            assert data["simulation"]["seed"] == 12345
        finally:
            os.unlink(variant_path)
