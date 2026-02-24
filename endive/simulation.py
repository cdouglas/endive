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
    ):
        self._config = config
        self._stats = Statistics(output_path=output_path)
        self._progress_path = progress_path
        self._start_wall: float = 0.0
        self._rng: Optional[np.random.RandomState] = None

    def run(self) -> Statistics:
        """Run simulation and return collected statistics."""
        if self._config.duration_ms <= 0:
            return self._stats

        if self._config.seed is not None:
            np.random.seed(self._config.seed)
            self._rng = np.random.RandomState(self._config.seed + 1)

        self._start_wall = time.time()
        env = simpy.Environment()
        env.process(self._run_workload(env))
        if self._progress_path:
            env.process(self._progress_reporter(env))
        env.run(until=self._config.duration_ms)

        self._stats.close()
        # Clean up progress file
        if self._progress_path:
            try:
                os.unlink(self._progress_path)
            except FileNotFoundError:
                pass
        return self._stats

    def _progress_reporter(
        self, env: simpy.Environment, interval_ms: float = 60_000,
    ) -> Generator:
        """Periodically write progress to a JSON file."""
        while True:
            yield env.timeout(interval_ms)
            progress = {
                "sim_time_ms": env.now,
                "duration_ms": self._config.duration_ms,
                "pct": round(env.now / self._config.duration_ms * 100, 1),
                "committed": self._stats.committed,
                "aborted": self._stats.aborted,
                "wall_clock_s": round(time.time() - self._start_wall, 2),
            }
            tmp = self._progress_path + ".tmp"
            try:
                with open(tmp, "w") as f:
                    json.dump(progress, f)
                os.replace(tmp, self._progress_path)
            except OSError:
                pass

    def _run_workload(self, env: simpy.Environment) -> Generator:
        """Generate transactions and launch them as SimPy processes."""
        for delay, txn in self._config.workload.generate():
            yield env.timeout(delay)
            env.process(self._execute_transaction(env, txn))

    def _execute_transaction(
        self,
        env: simpy.Environment,
        txn: Transaction,
    ) -> Generator:
        """Execute a single transaction and record its result."""
        gen = txn.execute(
            self._config.catalog,
            self._config.storage_provider,
            self._config.conflict_detector,
            self._config.max_retries,
            self._config.ml_append_mode,
        )

        result = yield from self._drive_generator(env, gen)
        self._stats.record_transaction(result)

    @staticmethod
    def _drive_generator(
        env: simpy.Environment,
        gen: Generator[float, None, TransactionResult],
    ) -> Generator:
        """Bridge a latency-yielding generator with SimPy timeouts.

        The transaction generators yield bare floats (latency in ms).
        This method converts each float into a SimPy timeout event,
        driving the generator to completion.

        Returns the generator's return value (TransactionResult).
        """
        try:
            latency = next(gen)
        except StopIteration as e:
            return e.value

        while True:
            yield env.timeout(latency)
            try:
                latency = gen.send(None)
            except StopIteration as e:
                return e.value
