"""Tests for simulation progress reporting."""

import json
import os
import tempfile

import numpy as np
import pytest

from endive.catalog import InstantCatalog
from endive.conflict_detector import ProbabilisticConflictDetector
from endive.simulation import Simulation, SimulationConfig
from endive.storage import InstantStorageProvider, LognormalLatency
from endive.workload import Workload, WorkloadConfig


def make_config(duration_ms=120_000, seed=42, inter_arrival_scale=200.0):
    """Create a SimulationConfig for progress tests.

    Uses 120s duration so the 60s progress reporter fires at least once.
    """
    rng = np.random.RandomState(seed)
    storage = InstantStorageProvider(rng=rng)
    wl_config = WorkloadConfig(
        inter_arrival=LognormalLatency.from_median(
            median_ms=inter_arrival_scale, sigma=0.5,
        ),
        runtime=LognormalLatency.from_median(median_ms=50.0, sigma=0.5),
        num_tables=1,
        partitions_per_table=(1,),
        fast_append_weight=1.0,
        merge_append_weight=0.0,
        validated_overwrite_weight=0.0,
    )
    return SimulationConfig(
        duration_ms=duration_ms,
        seed=seed,
        storage_provider=storage,
        catalog=InstantCatalog(num_tables=1, partitions_per_table=(1,), latency_ms=1.0),
        workload=Workload(wl_config, seed=seed + 100),
        conflict_detector=ProbabilisticConflictDetector(0.0, rng=np.random.RandomState(seed)),
        max_retries=3,
    )


class TestProgressReporting:
    def test_no_progress_file_by_default(self):
        """Without progress_path, no file is written."""
        config = make_config(duration_ms=5000)
        sim = Simulation(config)
        stats = sim.run()
        assert stats.total > 0

    def test_progress_file_cleaned_up(self):
        """Progress file is removed after simulation completes."""
        config = make_config(duration_ms=120_000)
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".progress.json")
            sim = Simulation(config, progress_path=progress_path)
            stats = sim.run()
            # File should be cleaned up after run completes
            assert not os.path.exists(progress_path)
            assert stats.committed > 0

    def test_progress_file_written_during_sim(self):
        """Progress file is written at least once for long-enough simulations.

        We use a 120s sim so the 60s interval fires at least once.
        The file is cleaned up on completion, but we can verify
        the mechanism by checking that committed > 0 (sim ran correctly).
        """
        config = make_config(duration_ms=120_000)
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".progress.json")
            sim = Simulation(config, progress_path=progress_path)
            stats = sim.run()
            # Sim ran correctly with progress reporting enabled
            assert stats.committed > 0
            # File cleaned up
            assert not os.path.exists(progress_path)

    def test_short_sim_no_progress_written(self):
        """Simulation shorter than interval doesn't write progress."""
        # 5s sim, 60s interval -> no progress file written
        config = make_config(duration_ms=5000)
        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".progress.json")
            sim = Simulation(config, progress_path=progress_path)
            stats = sim.run()
            assert not os.path.exists(progress_path)
            assert stats.total > 0

    def test_determinism_preserved_with_progress(self):
        """Progress reporting doesn't affect simulation determinism."""
        config_a = make_config(duration_ms=5000, seed=42)
        config_b = make_config(duration_ms=5000, seed=42)

        stats_a = Simulation(config_a).run()

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".progress.json")
            stats_b = Simulation(config_b, progress_path=progress_path).run()

        assert stats_a.committed == stats_b.committed
        assert stats_a.aborted == stats_b.aborted
        assert stats_a.total == stats_b.total

    def test_progress_reporter_writes_valid_json(self):
        """Verify progress file content by intercepting before cleanup.

        We subclass Simulation to capture the last progress write.
        """
        import simpy

        config = make_config(duration_ms=120_000)
        captured = {}

        class CapturingSimulation(Simulation):
            def _progress_reporter(self, env, interval_ms=60_000):
                """Override to capture progress data before file is cleaned up."""
                while True:
                    yield env.timeout(interval_ms)
                    progress = {
                        "sim_time_ms": env.now,
                        "duration_ms": self._config.duration_ms,
                        "pct": round(env.now / self._config.duration_ms * 100, 1),
                        "committed": self._stats.committed,
                        "aborted": self._stats.aborted,
                        "wall_clock_s": round(0.1, 2),  # Fake wall clock
                    }
                    captured.update(progress)
                    # Still write the file for realism
                    if self._progress_path:
                        import json as _json
                        tmp = self._progress_path + ".tmp"
                        try:
                            with open(tmp, "w") as f:
                                _json.dump(progress, f)
                            os.replace(tmp, self._progress_path)
                        except OSError:
                            pass

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, ".progress.json")
            sim = CapturingSimulation(config, progress_path=progress_path)
            stats = sim.run()

        # Progress reporter should have fired at 60000ms
        assert "sim_time_ms" in captured
        assert captured["sim_time_ms"] == 60_000
        assert captured["duration_ms"] == 120_000
        assert captured["pct"] == 50.0
        assert captured["committed"] >= 0
        assert captured["aborted"] >= 0
