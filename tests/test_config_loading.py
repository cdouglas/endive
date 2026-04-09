"""Tests for config loading per SPEC.md §7.

Tests:
- load_simulation_config: valid TOML → SimulationConfig
- Validation: invalid configs raise ConfigurationError
- Component wiring: storage, catalog, workload, conflict detector
- Defaults: operation weights, retries, manifest list mode
- Seed handling: from config, override, None
"""

import os
import tempfile

import numpy as np
import pytest

from endive.catalog import CASCatalog, InstantCatalog
from endive.config import (
    ConfigurationError,
    compute_experiment_hash,
    load_simulation_config,
    validate_config,
)
from endive.conflict_detector import (
    PartitionOverlapConflictDetector,
    ProbabilisticConflictDetector,
)
from endive.simulation import SimulationConfig
from endive.storage import InstantStorageProvider, S3ExpressStorageProvider
from endive.workload import Workload, ZipfTableSelector, UniformTableSelector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_toml(content: str) -> str:
    """Write TOML content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".toml")
    os.write(fd, content.encode())
    os.close(fd)
    return path


MINIMAL_CONFIG = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""

MULTI_TABLE_CONFIG = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 200.0
runtime.mean = 100000
runtime.sigma = 1.2
operation_types.fast_append = 0.5
operation_types.validated_overwrite = 0.5
real_conflict_probability = 0.3

[storage]
provider = "instant"
"""

S3X_CONFIG = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 10
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 180000
runtime.sigma = 1.5

[storage]
provider = "s3x"
"""


# ---------------------------------------------------------------------------
# load_simulation_config
# ---------------------------------------------------------------------------

class TestLoadSimulationConfig:
    def test_minimal_config(self):
        """Minimal valid config produces SimulationConfig."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config, SimulationConfig)
            assert config.duration_ms == 5000
            assert config.seed == 42
            assert config.max_retries == 3
        finally:
            os.unlink(path)

    def test_returns_all_components(self):
        """Config has all required components wired."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert config.storage_provider is not None
            assert config.catalog is not None
            assert config.workload is not None
            assert config.conflict_detector is not None
        finally:
            os.unlink(path)

    def test_instant_provider_creates_instant_catalog(self):
        """provider=instant creates InstantCatalog."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.catalog, InstantCatalog)
        finally:
            os.unlink(path)

    def test_s3x_provider_creates_cas_catalog(self):
        """provider=s3x creates CASCatalog."""
        path = write_toml(S3X_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.catalog, CASCatalog)
            assert isinstance(config.storage_provider, S3ExpressStorageProvider)
        finally:
            os.unlink(path)

    def test_seed_override(self):
        """seed_override replaces config file seed."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path, seed_override=999)
            assert config.seed == 999
        finally:
            os.unlink(path)

    def test_no_seed_in_config(self):
        """Config without seed uses None."""
        config_str = """\
[simulation]
duration_ms = 5000

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert config.seed is None
        finally:
            os.unlink(path)

    def test_multi_table(self):
        """Multi-table config builds correct topology."""
        path = write_toml(MULTI_TABLE_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config, SimulationConfig)
            # Workload should be configured for 5 tables
            assert isinstance(config.workload, Workload)
        finally:
            os.unlink(path)

    def test_mixed_operation_types(self):
        """Mixed operation types flow through to workload."""
        path = write_toml(MULTI_TABLE_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.conflict_detector,
                            ProbabilisticConflictDetector)
        finally:
            os.unlink(path)

    def test_manifest_list_mode_rewrite(self):
        """Default manifest_list_mode is rewrite → ml_append_mode=False."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert config.ml_append_mode is False
        finally:
            os.unlink(path)

    def test_manifest_list_mode_append(self):
        """manifest_list_mode=append → ml_append_mode=True."""
        config_str = MINIMAL_CONFIG + '\nmanifest_list_mode = "append"\n'
        # manifest_list_mode is in [transaction] section
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0
manifest_list_mode = "append"

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert config.ml_append_mode is True
        finally:
            os.unlink(path)

    def test_default_max_retries(self):
        """Default max_retries is 10 when not specified."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert config.max_retries == 10
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_invalid_num_tables(self):
        """num_tables <= 0 raises ConfigurationError."""
        config_str = """\
[simulation]
duration_ms = 5000

[catalog]
num_tables = 0

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="num_tables"):
                load_simulation_config(path)
        finally:
            os.unlink(path)

    def test_invalid_provider(self):
        """Unknown provider raises ConfigurationError."""
        config_str = """\
[simulation]
duration_ms = 5000

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "nonexistent"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="provider"):
                load_simulation_config(path)
        finally:
            os.unlink(path)

    def test_invalid_conflict_probability(self):
        """real_conflict_probability outside [0,1] raises error."""
        config_str = """\
[simulation]
duration_ms = 5000

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0
real_conflict_probability = 1.5

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="real_conflict_probability"):
                load_simulation_config(path)
        finally:
            os.unlink(path)

    def test_negative_operation_weight(self):
        """Negative operation weight raises error."""
        config_str = """\
[simulation]
duration_ms = 5000

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0
operation_types.fast_append = -0.5

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="fast_append"):
                load_simulation_config(path)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Validation (existing validate_config function)
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_valid_minimal(self):
        """Minimal valid config has no errors."""
        cfg = {
            "simulation": {"duration_ms": 5000},
            "catalog": {"num_tables": 1},
            "transaction": {
                "inter_arrival": {"distribution": "exponential", "scale": 100},
                "runtime": {"mean": 50000},
            },
            "storage": {"provider": "instant"},
        }
        errors, warnings = validate_config(cfg)
        assert len(errors) == 0

    def test_invalid_catalog_mode(self):
        """Invalid catalog mode produces error."""
        cfg = {
            "catalog": {"num_tables": 1, "mode": "invalid"},
        }
        errors, _ = validate_config(cfg)
        assert any("mode" in e for e in errors)

    def test_unknown_operation_type(self):
        """Unknown operation type produces error."""
        cfg = {
            "catalog": {"num_tables": 1},
            "transaction": {"operation_types": {"unknown_op": 0.5}},
        }
        errors, _ = validate_config(cfg)
        assert any("unknown_op" in e.lower() or "Unknown" in e for e in errors)

    def test_weight_normalization_warning(self):
        """Weights not summing to 1.0 produces warning."""
        cfg = {
            "catalog": {"num_tables": 1},
            "transaction": {
                "operation_types": {
                    "fast_append": 0.5,
                    "validated_overwrite": 0.3,
                },
            },
        }
        _, warnings = validate_config(cfg)
        assert any("normalized" in w for w in warnings)


# ---------------------------------------------------------------------------
# Component building
# ---------------------------------------------------------------------------

class TestComponentBuilding:
    def test_probabilistic_conflict_detector(self):
        """Default config creates ProbabilisticConflictDetector."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.conflict_detector,
                            ProbabilisticConflictDetector)
        finally:
            os.unlink(path)

    def test_partition_overlap_detector(self):
        """partition.enabled creates PartitionOverlapConflictDetector."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"

[partition]
enabled = true
num_partitions = 10
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.conflict_detector,
                            PartitionOverlapConflictDetector)
        finally:
            os.unlink(path)

    def test_fixed_inter_arrival(self):
        """fixed distribution creates proper workload."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
inter_arrival.distribution = "fixed"
inter_arrival.value = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.workload, Workload)
        finally:
            os.unlink(path)

    def test_inlined_metadata_config(self):
        """table_metadata_inlined=true creates InstantCatalog with inlining params."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 2
table_metadata_inlined = true
initial_partition_size_bytes = 1024
commit_growth_bytes = 50
backend = "service"

[catalog.partition]
num_partitions = 5

[catalog.service]
provider = "instant"
latency_ms = 2.0
latency_per_kib_ms = 0.5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[partition]
enabled = true

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.catalog, InstantCatalog)
            assert config.catalog.metadata_inlined is True
            # 2 tables * 5 partitions * 1024 bytes = 10240
            assert config.catalog.catalog_size_bytes == 10240
            assert config.metadata_inlined is True
        finally:
            os.unlink(path)

    def test_inlined_metadata_default_false(self):
        """Without table_metadata_inlined, defaults to False."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            assert config.metadata_inlined is False
        finally:
            os.unlink(path)

    def test_inlined_latency_per_kib_flows_to_catalog(self):
        """latency_per_kib_ms affects effective latency on read."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1
table_metadata_inlined = true
initial_partition_size_bytes = 2048
backend = "service"

[catalog.service]
provider = "instant"
latency_ms = 1.0
latency_per_kib_ms = 1.0

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            cat = config.catalog
            # Effective latency = 1.0 + (2048/1024) * 1.0 = 3.0. First yield = half.
            gen = cat.read()
            latency = next(gen)
            assert latency == pytest.approx(1.5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# End-to-end: load → run → verify
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_load_and_run(self):
        """Loaded config runs simulation to completion."""
        from endive.simulation import Simulation

        # Use short runtime so transactions complete within duration
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 1

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 200.0
runtime.mean = 50.0
runtime.sigma = 0.5

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            stats = Simulation(config).run()
            assert stats.total > 0
            assert stats.committed > 0
        finally:
            os.unlink(path)

    def test_deterministic_load_and_run(self):
        """Same config file + same seed → identical results."""
        from endive.simulation import Simulation

        path = write_toml(MINIMAL_CONFIG)
        try:
            config_a = load_simulation_config(path)
            config_b = load_simulation_config(path)
            stats_a = Simulation(config_a).run()
            stats_b = Simulation(config_b).run()
            assert stats_a.committed == stats_b.committed
            assert stats_a.aborted == stats_b.aborted
            for a, b in zip(stats_a.transactions, stats_b.transactions):
                assert a.txn_id == b.txn_id
                assert a.commit_time_ms == b.commit_time_ms
        finally:
            os.unlink(path)

    def test_existing_experiment_config(self):
        """Can load an actual experiment config file."""
        from endive.simulation import Simulation

        config_path = "/app/experiment_configs/exp1_fa_baseline.toml"
        if not os.path.exists(config_path):
            pytest.skip("Experiment config not found")

        config = load_simulation_config(config_path, seed_override=42)
        # Just verify it builds without error
        assert isinstance(config, SimulationConfig)
        assert config.seed == 42

    def test_inlined_metadata_simulation(self):
        """Full simulation with inlined metadata completes and shows latency growth."""
        from endive.simulation import Simulation

        config_str = """\
[simulation]
duration_ms = 10000
seed = 42

[catalog]
num_tables = 1
table_metadata_inlined = true
initial_partition_size_bytes = 0
commit_growth_bytes = 10240
backend = "service"

[catalog.partition]
num_partitions = 1

[catalog.service]
provider = "instant"
latency_ms = 0.0
latency_per_kib_ms = 1.0

[transaction]
retry = 10
tables_per_txn = 1
inter_arrival.distribution = "exponential"
inter_arrival.scale = 50.0
runtime.mean = 10.0
runtime.sigma = 0.5
operation_types.fast_append = 1.0
operation_types.validated_overwrite = 0.0

[partition]
enabled = true
partitions_per_txn = 1

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert config.metadata_inlined is True
            stats = Simulation(config).run()
            assert stats.total > 0
            assert stats.committed > 0
            # Catalog should have grown
            assert config.catalog.catalog_size_bytes > 0
            # Verify later commits have higher CAS latency than earlier commits
            txns = [t for t in stats.transactions if t.status.name == "committed"]
            if len(txns) >= 4:
                early = txns[:len(txns) // 4]
                late = txns[-len(txns) // 4:]
                avg_early = sum(t.commit_time_ms - t.submit_time_ms for t in early) / len(early)
                avg_late = sum(t.commit_time_ms - t.submit_time_ms for t in late) / len(late)
                # Late transactions should have higher latency due to catalog growth
                assert avg_late > avg_early, (
                    f"Expected latency growth: early={avg_early:.1f}ms, late={avg_late:.1f}ms"
                )
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Experiment hash
# ---------------------------------------------------------------------------

class TestExperimentHash:
    def test_plots_section_excluded_from_hash(self):
        """Adding [plots] section does not change experiment hash."""
        config_without_plots = {
            "simulation": {"duration_ms": 3600000, "seed": 42},
            "catalog": {"num_tables": 1},
            "transaction": {
                "inter_arrival": {"scale": 100.0},
                "runtime": {"mean": 180000},
            },
            "storage": {"provider": "s3"},
            "experiment": {"label": "test"},
        }
        config_with_plots = {
            **config_without_plots,
            "plots": {
                "output_dir": "plots/test",
                "graphs": [
                    {"type": "latency_vs_throughput"},
                    {"type": "success_rate_vs_throughput", "group_by": "num_tables"},
                ],
            },
        }
        hash_without = compute_experiment_hash(config_without_plots)
        hash_with = compute_experiment_hash(config_with_plots)
        assert hash_without == hash_with

    def test_seed_excluded_from_hash(self):
        """Different seeds produce the same hash."""
        base = {
            "simulation": {"duration_ms": 3600000},
            "catalog": {"num_tables": 1},
            "transaction": {"inter_arrival": {"scale": 100.0}},
            "storage": {"provider": "s3"},
        }
        config_a = {**base, "simulation": {**base["simulation"], "seed": 42}}
        config_b = {**base, "simulation": {**base["simulation"], "seed": 99}}
        assert compute_experiment_hash(config_a) == compute_experiment_hash(config_b)

    def test_param_change_changes_hash(self):
        """Changing a simulation parameter changes the hash."""
        base = {
            "simulation": {"duration_ms": 3600000},
            "catalog": {"num_tables": 1},
            "transaction": {"inter_arrival": {"scale": 100.0}},
            "storage": {"provider": "s3"},
        }
        config_a = {**base}
        config_b = {**base, "transaction": {"inter_arrival": {"scale": 200.0}}}
        assert compute_experiment_hash(config_a) != compute_experiment_hash(config_b)

    def test_table_selection_changes_hash(self):
        """Adding table_selection changes experiment hash."""
        base = {
            "simulation": {"duration_ms": 3600000},
            "catalog": {"num_tables": 10},
            "transaction": {"inter_arrival": {"scale": 100.0}},
            "storage": {"provider": "s3"},
        }
        config_zipf = {
            **base,
            "transaction": {
                **base["transaction"],
                "table_selection": {"distribution": "zipf", "zipf_alpha": 1.5},
            },
        }
        assert compute_experiment_hash(base) != compute_experiment_hash(config_zipf)


# ---------------------------------------------------------------------------
# Table selection config
# ---------------------------------------------------------------------------

class TestTableSelectionConfig:
    """Tests for [transaction.table_selection] config parsing."""

    def test_default_uniform(self):
        """No table_selection section → uniform selector."""
        path = write_toml(MINIMAL_CONFIG)
        try:
            config = load_simulation_config(path)
            # Workload defaults to UniformTableSelector when no table_selector
            assert isinstance(config.workload._table_selector, UniformTableSelector)
        finally:
            os.unlink(path)

    def test_zipf_distribution(self):
        """distribution=zipf creates ZipfTableSelector with correct alpha."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 10

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "zipf"
zipf_alpha = 2.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            selector = config.workload._table_selector
            assert isinstance(selector, ZipfTableSelector)
            assert selector._alpha == 2.0
        finally:
            os.unlink(path)

    def test_zipf_default_alpha(self):
        """Zipf without zipf_alpha uses default 1.5."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "zipf"

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            selector = config.workload._table_selector
            assert isinstance(selector, ZipfTableSelector)
            assert selector._alpha == 1.5
        finally:
            os.unlink(path)

    def test_explicit_uniform(self):
        """distribution=uniform creates UniformTableSelector."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "uniform"

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            config = load_simulation_config(path)
            assert isinstance(config.workload._table_selector, UniformTableSelector)
        finally:
            os.unlink(path)

    def test_invalid_distribution(self):
        """Invalid distribution raises ConfigurationError."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "gaussian"

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="table_selection.distribution"):
                load_simulation_config(path)
        finally:
            os.unlink(path)

    def test_zipf_alpha_zero(self):
        """zipf_alpha <= 0 raises ConfigurationError."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "zipf"
zipf_alpha = 0.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="zipf_alpha"):
                load_simulation_config(path)
        finally:
            os.unlink(path)

    def test_zipf_alpha_negative(self):
        """Negative zipf_alpha raises ConfigurationError."""
        config_str = """\
[simulation]
duration_ms = 5000
seed = 42

[catalog]
num_tables = 5

[transaction]
retry = 3
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
runtime.mean = 50000
runtime.sigma = 1.0

[transaction.table_selection]
distribution = "zipf"
zipf_alpha = -1.0

[storage]
provider = "instant"
"""
        path = write_toml(config_str)
        try:
            with pytest.raises(ConfigurationError, match="zipf_alpha"):
                load_simulation_config(path)
        finally:
            os.unlink(path)
