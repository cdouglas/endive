"""Simulation runner per SPEC.md §6.

Coordinates workload generation, transaction execution, and statistics
collection using SimPy as the discrete-event simulation engine.

Key types:
- SimulationConfig: Complete simulation configuration (frozen)
- Statistics: Aggregate counters and per-transaction results
- Simulation: Main runner that bridges generators with SimPy

The simulation runner is the ONLY place SimPy is used. All other
components (StorageProvider, Catalog, Transaction, Workload) yield
bare floats representing latencies in milliseconds.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import simpy

from endive.catalog import Catalog
from endive.storage import StorageProvider
from endive.transaction import (
    ConflictDetector,
    Transaction,
    TransactionResult,
    TransactionStatus,
)
from endive.workload import Workload


# ---------------------------------------------------------------------------
# SimulationConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulation configuration.

    All components are fully constructed before simulation starts.
    The config is frozen to prevent accidental mutation during a run.
    """
    duration_ms: float
    seed: Optional[int]

    # Components (constructed by config loader)
    storage_provider: StorageProvider
    catalog: Catalog
    workload: Workload
    conflict_detector: ConflictDetector

    # Transaction limits
    max_retries: int = 10
    ml_append_mode: bool = False

    # Backoff configuration
    backoff_enabled: bool = False
    backoff_base_ms: float = 10.0
    backoff_multiplier: float = 2.0
    backoff_max_ms: float = 5000.0
    backoff_jitter: float = 0.1


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

# Arrow schema for parquet output (shared between streaming and in-memory paths)
_ARROW_SCHEMA = pa.schema([
    ("txn_id", pa.int64()),
    ("t_submit", pa.int64()),
    ("t_runtime", pa.int64()),
    ("t_commit", pa.int64()),
    ("commit_latency", pa.int64()),
    ("total_latency", pa.int64()),
    ("n_retries", pa.int32()),
    ("status", pa.string()),
    ("operation_type", pa.string()),
    ("abort_reason", pa.string()),
    ("manifest_list_reads", pa.int32()),
    ("manifest_list_writes", pa.int32()),
    ("manifest_file_reads", pa.int32()),
    ("manifest_file_writes", pa.int32()),
    ("catalog_read_ms", pa.float32()),
    ("per_attempt_io_ms", pa.float32()),
    ("conflict_io_ms", pa.float32()),
    ("catalog_commit_ms", pa.float32()),
    ("event_count", pa.int32()),
])


def _result_to_row(r: TransactionResult) -> dict:
    """Convert a TransactionResult to a dict matching the Arrow schema."""
    is_committed = r.status == TransactionStatus.COMMITTED
    return {
        "txn_id": r.txn_id,
        "t_submit": int(round(r.commit_time_ms - r.total_latency_ms))
            if is_committed
            else int(round(r.abort_time_ms - r.total_latency_ms)),
        "t_runtime": int(round(r.runtime_ms)),
        "t_commit": int(round(r.commit_time_ms)) if is_committed else -1,
        "commit_latency": int(round(r.commit_latency_ms)) if is_committed else -1,
        "total_latency": int(round(r.total_latency_ms)),
        "n_retries": r.total_retries,
        "status": "committed" if is_committed else "aborted",
        "operation_type": r.operation_type,
        "abort_reason": r.abort_reason,
        "manifest_list_reads": r.manifest_list_reads,
        "manifest_list_writes": r.manifest_list_writes,
        "manifest_file_reads": r.manifest_file_reads,
        "manifest_file_writes": r.manifest_file_writes,
        "catalog_read_ms": round(r.catalog_read_ms, 2),
        "per_attempt_io_ms": round(r.per_attempt_io_ms, 2),
        "conflict_io_ms": round(r.conflict_io_ms, 2),
        "catalog_commit_ms": round(r.catalog_commit_ms, 2),
        "event_count": r.event_count,
    }


def _rows_to_arrow_table(rows: list[dict]) -> pa.Table:
    """Convert a list of row dicts to a pyarrow Table with the shared schema."""
    arrays = {}
    for field in _ARROW_SCHEMA:
        arrays[field.name] = pa.array(
            [row[field.name] for row in rows],
            type=field.type,
        )
    return pa.table(arrays, schema=_ARROW_SCHEMA)


class Statistics:
    """Collected simulation statistics with optional streaming parquet export.

    When output_path is provided, results are written incrementally to parquet
    in row groups of buffer_size, keeping memory at O(buffer_size) per process.

    When output_path is None (tests, in-memory use), results are accumulated
    in a list as before.
    """

    def __init__(
        self,
        output_path: str | None = None,
        buffer_size: int = 1000,
    ):
        self._output_path = output_path
        self._buffer_size = buffer_size
        self._buffer: list[dict] = []
        self._writer: pq.ParquetWriter | None = None

        # In-memory fallback (when no output_path)
        self.transactions: list[TransactionResult] = []

        # Aggregate counters
        self.committed: int = 0
        self.aborted: int = 0
        self.total_retries: int = 0
        self.validation_exceptions: int = 0

        # I/O counters
        self.manifest_list_reads: int = 0
        self.manifest_list_writes: int = 0
        self.manifest_file_reads: int = 0
        self.manifest_file_writes: int = 0

    def record_transaction(self, result: TransactionResult) -> None:
        """Record completed transaction result."""
        # Update aggregate counters
        if result.status == TransactionStatus.COMMITTED:
            self.committed += 1
        elif result.status == TransactionStatus.ABORTED:
            self.aborted += 1
            if result.abort_reason == "validation_exception":
                self.validation_exceptions += 1

        self.total_retries += result.total_retries
        self.manifest_list_reads += result.manifest_list_reads
        self.manifest_list_writes += result.manifest_list_writes
        self.manifest_file_reads += result.manifest_file_reads
        self.manifest_file_writes += result.manifest_file_writes

        if self._output_path:
            # Streaming: buffer row dict, flush when full
            self._buffer.append(_result_to_row(result))
            if len(self._buffer) >= self._buffer_size:
                self._flush()
        else:
            # In-memory: keep TransactionResult for test access
            self.transactions.append(result)

    def _flush(self) -> None:
        """Write buffered rows as a parquet row group."""
        if not self._buffer:
            return
        table = _rows_to_arrow_table(self._buffer)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._output_path, _ARROW_SCHEMA, compression="snappy",
            )
        self._writer.write_table(table)
        self._buffer.clear()

    def close(self) -> None:
        """Flush remaining buffer and close the parquet writer."""
        self._flush()
        if self._writer:
            self._writer.close()
            self._writer = None

    @property
    def total(self) -> int:
        """Total transactions recorded."""
        return self.committed + self.aborted

    @property
    def success_rate(self) -> float:
        """Fraction of transactions that committed."""
        if self.total == 0:
            return 0.0
        return self.committed / self.total

    def to_dataframe(self) -> pd.DataFrame:
        """Export transactions to DataFrame for analysis.

        If streaming to a file, reads back the written parquet.
        If in-memory, converts the accumulated transactions list.
        """
        if self._output_path:
            # Ensure everything is flushed
            self.close()
            if self.total == 0:
                return pd.DataFrame()
            return pd.read_parquet(self._output_path)

        if not self.transactions:
            return pd.DataFrame()

        rows = [_result_to_row(r) for r in self.transactions]
        table = _rows_to_arrow_table(rows)
        return table.to_pandas()

    def export_parquet(self, path: str) -> None:
        """Export to parquet file.

        If already streaming to output_path, this closes the writer.
        If in-memory, writes a new file at path.
        """
        if self._output_path:
            self.close()
            return

        if not self.transactions:
            # Write empty file with correct schema
            pq.write_table(
                pa.table({f.name: pa.array([], type=f.type) for f in _ARROW_SCHEMA},
                         schema=_ARROW_SCHEMA),
                path, compression="snappy",
            )
            return

        rows = [_result_to_row(r) for r in self.transactions]
        table = _rows_to_arrow_table(rows)
        pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# DES engine profiling
# ---------------------------------------------------------------------------

class _CountingEnvironment(simpy.Environment):
    """SimPy environment that counts discrete events processed."""

    def __init__(self, initial_time: float = 0):
        super().__init__(initial_time)
        self.event_count: int = 0

    def step(self) -> None:
        self.event_count += 1
        super().step()

    @property
    def queue_depth(self) -> int:
        return len(self._queue)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class Simulation:
    """Main simulation runner.

    Coordinates workload generation, transaction execution,
    and statistics collection via SimPy discrete-event simulation.

    Usage:
        config = SimulationConfig(...)
        sim = Simulation(config)
        stats = sim.run()
        stats.export_parquet("results.parquet")

    For streaming export (lower memory):
        sim = Simulation(config, output_path="results.parquet")
        stats = sim.run()  # results written incrementally
    """

    def __init__(
        self,
        config: SimulationConfig,
        output_path: str | None = None,
        progress_path: str | None = None,
        profile: bool = False,
    ):
        self._config = config
        self._stats = Statistics(output_path=output_path)
        self._progress_path = progress_path
        self._profile = profile
        self._start_wall: float = 0.0
        self._rng: Optional[np.random.RandomState] = None

        # Active process tracking (for profiling)
        self._active_processes: int = 0
        self._peak_processes: int = 0

        # Profile samples (collected by _progress_reporter)
        self._profile_samples: list[dict] = []
        self._env: Optional[_CountingEnvironment] = None

    def run(self) -> Statistics:
        """Run simulation and return collected statistics."""
        if self._config.duration_ms <= 0:
            return self._stats

        if self._config.seed is not None:
            np.random.seed(self._config.seed)
            self._rng = np.random.RandomState(self._config.seed + 1)

        self._start_wall = time.time()
        env = _CountingEnvironment()
        self._env = env
        env.process(self._run_workload(env))
        if self._progress_path:
            env.process(self._progress_reporter(env))
        env.run(until=self._config.duration_ms)

        self._stats.close()

        # Write profile output before cleaning up progress
        if self._profile and self._progress_path:
            self._write_profile_json(env)

        # Clean up progress file
        if self._progress_path:
            try:
                os.unlink(self._progress_path)
            except FileNotFoundError:
                pass
        return self._stats

    def _write_profile_json(self, env: _CountingEnvironment) -> None:
        """Write .profile.json with DES engine profiling data."""
        output_dir = os.path.dirname(self._progress_path)
        profile_path = os.path.join(output_dir, ".profile.json")

        wall_total = time.time() - self._start_wall
        des_total = env.event_count

        # Compute summary from samples
        if self._profile_samples:
            rates = [s["des_rate"] for s in self._profile_samples if s["des_rate"] > 0]
            depths = [s["queue_depth"] for s in self._profile_samples]
            speeds = [s["sim_speed"] for s in self._profile_samples if s["sim_speed"] > 0]
            slow_intervals = sum(1 for s in self._profile_samples if s["sim_speed"] < 1.0)
        else:
            rates = []
            depths = []
            speeds = []
            slow_intervals = 0

        summary = {
            "des_events_total": des_total,
            "wall_clock_s": round(wall_total, 2),
            "des_rate_mean": round(sum(rates) / len(rates), 1) if rates else 0,
            "des_rate_min": round(min(rates), 1) if rates else 0,
            "des_rate_max": round(max(rates), 1) if rates else 0,
            "queue_depth_max": max(depths) if depths else 0,
            "queue_depth_mean": round(sum(depths) / len(depths), 1) if depths else 0,
            "peak_processes": self._peak_processes,
            "sim_speed_min": round(min(speeds), 2) if speeds else 0,
            "slow_intervals": slow_intervals,
        }

        profile = {"summary": summary, "samples": self._profile_samples}

        try:
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)
        except OSError:
            pass

    def _progress_reporter(
        self, env: _CountingEnvironment, interval_ms: float = 60_000,
    ) -> Generator:
        """Periodically write progress to a JSON file."""
        last_wall = self._start_wall
        last_events = 0
        last_sim_time = 0.0

        while True:
            yield env.timeout(interval_ms)
            now_wall = time.time()
            wall_delta = now_wall - last_wall
            event_delta = env.event_count - last_events
            sim_delta = env.now - last_sim_time
            des_rate = event_delta / wall_delta if wall_delta > 0 else 0
            # sim_speed: how many sim-seconds pass per wall-second
            sim_speed = (sim_delta / 1000.0) / wall_delta if wall_delta > 0 else 0

            progress = {
                "sim_time_ms": env.now,
                "duration_ms": self._config.duration_ms,
                "pct": round(env.now / self._config.duration_ms * 100, 1),
                "committed": self._stats.committed,
                "aborted": self._stats.aborted,
                "wall_clock_s": round(now_wall - self._start_wall, 2),
                "des_rate": round(des_rate, 1),
                "queue_depth": env.queue_depth,
            }
            tmp = self._progress_path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(progress, f)
                os.replace(tmp, self._progress_path)
            except OSError:
                pass

            # Collect profile sample if profiling enabled
            if self._profile:
                self._profile_samples.append({
                    "sim_time_ms": env.now,
                    "wall_clock_s": round(now_wall - self._start_wall, 2),
                    "wall_delta_s": round(wall_delta, 3),
                    "des_events": event_delta,
                    "des_rate": round(des_rate, 1),
                    "queue_depth": env.queue_depth,
                    "active_processes": self._active_processes,
                    "sim_speed": round(sim_speed, 2),
                })

            last_wall = now_wall
            last_events = env.event_count
            last_sim_time = env.now

    def _run_workload(self, env: simpy.Environment) -> Generator:
        """Generate transactions and launch them as SimPy processes."""
        for delay, txn in self._config.workload.generate():
            yield env.timeout(delay)
            env.process(self._execute_transaction(env, txn))

    def _execute_transaction(
        self,
        env: _CountingEnvironment,
        txn: Transaction,
    ) -> Generator:
        """Execute a single transaction and record its result."""
        self._active_processes += 1
        self._peak_processes = max(self._peak_processes, self._active_processes)

        gen = txn.execute(
            self._config.catalog,
            self._config.storage_provider,
            self._config.conflict_detector,
            self._config.max_retries,
            self._config.ml_append_mode,
        )

        try:
            result = yield from self._drive_generator(env, gen)
            self._stats.record_transaction(result)
        finally:
            self._active_processes -= 1

    @staticmethod
    def _drive_generator(
        env: simpy.Environment,
        gen: Generator[float, None, TransactionResult],
    ) -> Generator:
        """Bridge a latency-yielding generator with SimPy timeouts.

        The transaction generators yield bare floats (latency in ms).
        This method converts each float into a SimPy timeout event,
        driving the generator to completion.

        Returns the generator's return value (TransactionResult) with
        event_count attached when using a _CountingEnvironment.
        """
        has_counting = hasattr(env, 'event_count')
        event_start = env.event_count if has_counting else 0
        try:
            latency = next(gen)
        except StopIteration as e:
            result = e.value
            if has_counting and isinstance(result, TransactionResult):
                event_count = env.event_count - event_start
                result = dataclasses.replace(result, event_count=event_count)
            return result

        while True:
            yield env.timeout(latency)
            try:
                latency = gen.send(None)
            except StopIteration as e:
                result = e.value
                if has_counting and isinstance(result, TransactionResult):
                    event_count = env.event_count - event_start
                    result = dataclasses.replace(result, event_count=event_count)
                return result
