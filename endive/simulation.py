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

from dataclasses import dataclass, field
from typing import Generator, List, Optional

import numpy as np
import pandas as pd
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

@dataclass
class Statistics:
    """Collected simulation statistics.

    Records per-transaction results and maintains aggregate counters.
    Thread-safe within SimPy (single-threaded event loop).
    """
    transactions: List[TransactionResult] = field(default_factory=list)

    # Aggregate counters
    committed: int = 0
    aborted: int = 0
    total_retries: int = 0
    validation_exceptions: int = 0

    # I/O counters
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0

    def record_transaction(self, result: TransactionResult) -> None:
        """Record completed transaction result."""
        self.transactions.append(result)

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

        Column names are compatible with the existing analysis pipeline.
        """
        if not self.transactions:
            return pd.DataFrame()

        rows = []
        for r in self.transactions:
            rows.append({
                "txn_id": r.txn_id,
                "t_submit": int(round(r.commit_time_ms - r.total_latency_ms))
                    if r.status == TransactionStatus.COMMITTED
                    else int(round(r.abort_time_ms - r.total_latency_ms)),
                "t_commit": int(round(r.commit_time_ms))
                    if r.status == TransactionStatus.COMMITTED
                    else -1,
                "commit_latency": int(round(r.commit_latency_ms))
                    if r.status == TransactionStatus.COMMITTED
                    else -1,
                "total_latency": int(round(r.total_latency_ms)),
                "n_retries": r.total_retries,
                "status": "committed"
                    if r.status == TransactionStatus.COMMITTED
                    else "aborted",
                "abort_reason": r.abort_reason,
                "manifest_list_reads": r.manifest_list_reads,
                "manifest_list_writes": r.manifest_list_writes,
                "manifest_file_reads": r.manifest_file_reads,
                "manifest_file_writes": r.manifest_file_writes,
            })

        df = pd.DataFrame(rows)

        # Optimize dtypes
        int_cols = [
            "txn_id", "t_submit", "t_commit", "commit_latency",
            "total_latency",
        ]
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype("int64")

        small_int_cols = [
            "n_retries", "manifest_list_reads", "manifest_list_writes",
            "manifest_file_reads", "manifest_file_writes",
        ]
        for col in small_int_cols:
            if col in df.columns:
                df[col] = df[col].astype("int32")

        return df

    def export_parquet(self, path: str) -> None:
        """Export to parquet file."""
        df = self.to_dataframe()
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)


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
    """

    def __init__(self, config: SimulationConfig):
        self._config = config
        self._stats = Statistics()
        self._rng: Optional[np.random.RandomState] = None

    def run(self) -> Statistics:
        """Run simulation and return collected statistics."""
        if self._config.duration_ms <= 0:
            return self._stats

        if self._config.seed is not None:
            np.random.seed(self._config.seed)
            self._rng = np.random.RandomState(self._config.seed + 1)

        env = simpy.Environment()
        env.process(self._run_workload(env))
        env.run(until=self._config.duration_ms)

        return self._stats

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
