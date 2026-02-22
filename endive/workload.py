"""Workload generator per SPEC.md §4.

The Workload generates Transactions with encapsulated rate and parameters.
Topology (tables, partitions) comes from WorkloadConfig, NOT from Catalog.

Key types:
- WorkloadConfig: Immutable workload configuration
- TableSelector / PartitionSelector: ABCs for selection strategies
- UniformTableSelector / ZipfTableSelector: Concrete table selectors
- UniformPartitionSelector / ZipfPartitionSelector: Concrete partition selectors
- Workload: Transaction generator (the only public interface)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, FrozenSet, Generator, Optional, Tuple

import numpy as np

from endive.storage import LatencyDistribution
from endive.transaction import (
    FastAppendTransaction,
    MergeAppendTransaction,
    Transaction,
    ValidatedOverwriteTransaction,
)


# ---------------------------------------------------------------------------
# Table and Partition Selectors
# ---------------------------------------------------------------------------

class TableSelector(ABC):
    """Selects which tables a transaction touches."""

    @abstractmethod
    def select(
        self,
        n_tables: int,
        total_tables: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """Select read and write table sets.

        Returns:
            (tables_read, tables_written) where tables_written ⊆ tables_read
        """
        ...


class UniformTableSelector(TableSelector):
    """Uniform random table selection."""

    def __init__(self, write_fraction: float = 1.0):
        self._write_fraction = write_fraction

    def select(
        self,
        n_tables: int,
        total_tables: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        n = min(n_tables, total_tables)
        chosen = rng.choice(total_tables, size=n, replace=False)
        tables_read = frozenset(int(t) for t in chosen)

        n_write = max(1, int(n * self._write_fraction))
        if n_write >= n:
            tables_written = tables_read
        else:
            write_ids = rng.choice(sorted(tables_read), size=n_write, replace=False)
            tables_written = frozenset(int(t) for t in write_ids)

        return tables_read, tables_written


class ZipfTableSelector(TableSelector):
    """Zipf-distributed table selection (hot tables).

    Lower table IDs are selected more frequently.
    Alpha controls skew: higher alpha = more concentrated on table 0.
    """

    def __init__(self, alpha: float = 1.5, write_fraction: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._alpha = alpha
        self._write_fraction = write_fraction

    def select(
        self,
        n_tables: int,
        total_tables: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        n = min(n_tables, total_tables)
        pmf = _truncated_zipf_pmf(total_tables, self._alpha)
        chosen = rng.choice(total_tables, size=n, p=pmf, replace=False)
        tables_read = frozenset(int(t) for t in chosen)

        n_write = max(1, int(n * self._write_fraction))
        if n_write >= n:
            tables_written = tables_read
        else:
            write_ids = rng.choice(sorted(tables_read), size=n_write, replace=False)
            tables_written = frozenset(int(t) for t in write_ids)

        return tables_read, tables_written


class PartitionSelector(ABC):
    """Selects which partitions within a table are touched."""

    @abstractmethod
    def select(
        self,
        n_partitions: int,
        total_partitions: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """Select read and write partition sets.

        Returns:
            (partitions_read, partitions_written) where
            partitions_written ⊆ partitions_read
        """
        ...


class UniformPartitionSelector(PartitionSelector):
    """Uniform random partition selection."""

    def __init__(self, write_fraction: float = 1.0):
        self._write_fraction = write_fraction

    def select(
        self,
        n_partitions: int,
        total_partitions: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        n = min(n_partitions, total_partitions)
        chosen = rng.choice(total_partitions, size=n, replace=False)
        partitions_read = frozenset(int(p) for p in chosen)

        n_write = max(1, int(n * self._write_fraction))
        if n_write >= n:
            partitions_written = partitions_read
        else:
            write_ids = rng.choice(sorted(partitions_read), size=n_write, replace=False)
            partitions_written = frozenset(int(p) for p in write_ids)

        return partitions_read, partitions_written


class ZipfPartitionSelector(PartitionSelector):
    """Zipf-distributed partition selection (hot partitions)."""

    def __init__(self, alpha: float = 1.5, write_fraction: float = 1.0):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self._alpha = alpha
        self._write_fraction = write_fraction

    def select(
        self,
        n_partitions: int,
        total_partitions: int,
        rng: np.random.RandomState,
    ) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        n = min(n_partitions, total_partitions)
        pmf = _truncated_zipf_pmf(total_partitions, self._alpha)
        chosen = rng.choice(total_partitions, size=n, p=pmf, replace=False)
        partitions_read = frozenset(int(p) for p in chosen)

        n_write = max(1, int(n * self._write_fraction))
        if n_write >= n:
            partitions_written = partitions_read
        else:
            write_ids = rng.choice(sorted(partitions_read), size=n_write, replace=False)
            partitions_written = frozenset(int(p) for p in write_ids)

        return partitions_read, partitions_written


def _truncated_zipf_pmf(n: int, alpha: float) -> np.ndarray:
    """Compute truncated Zipf probability mass function.

    P(k) = (1/k^alpha) / sum(1/i^alpha for i in 1..n)
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, alpha)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# WorkloadConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadConfig:
    """Immutable workload configuration.

    Topology (num_tables, partitions_per_table) is owned by the Workload,
    NOT by the Catalog. This is a key design decision from SPEC.md §4.
    """
    # Required: arrival and runtime distributions
    inter_arrival: LatencyDistribution
    runtime: LatencyDistribution

    # Topology (fixed at simulation start)
    num_tables: int
    partitions_per_table: Tuple[int, ...]

    # Operation type weights (normalized internally)
    fast_append_weight: float = 0.7
    merge_append_weight: float = 0.2
    validated_overwrite_weight: float = 0.1

    # Table selection
    tables_per_txn: int = 1
    table_selector: Optional[TableSelector] = None  # None = uniform

    # Partition selection (None = no partition tracking)
    partitions_per_txn: Optional[int] = None
    partition_selector: Optional[PartitionSelector] = None  # None = uniform

    # MergeAppend parameter
    manifests_per_concurrent_commit: float = 1.5


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class Workload:
    """Transaction generator with encapsulated rate and parameters.

    The Workload knows the table/partition topology and configures
    Transactions accordingly. The rate at which transactions are
    produced and their parameters are not visible outside this class.

    Usage:
        for delay, txn in workload.generate():
            # delay is the inter-arrival time to wait
            # txn is a ready-to-execute Transaction
            yield sim.timeout(delay)
            sim.process(run_transaction(txn))
    """

    def __init__(
        self,
        config: WorkloadConfig,
        seed: Optional[int] = None,
    ):
        self._config = config
        self._rng = np.random.RandomState(seed)
        self._txn_counter = 0

        # Normalize operation type weights
        weights = np.array([
            config.fast_append_weight,
            config.merge_append_weight,
            config.validated_overwrite_weight,
        ])
        total = weights.sum()
        if total <= 0:
            raise ValueError("Operation type weights must sum to > 0")
        self._op_weights = weights / total

        # Default selectors
        self._table_selector = config.table_selector or UniformTableSelector()
        self._partition_selector = config.partition_selector or UniformPartitionSelector()

    @property
    def config(self) -> WorkloadConfig:
        return self._config

    def generate(self) -> Generator[Tuple[float, Transaction], None, None]:
        """Generate (inter_arrival_delay, transaction) pairs.

        Yields pairs of (delay_ms, Transaction) indefinitely.
        The simulation runner should wait delay_ms before launching
        the transaction.

        This is the ONLY interface for obtaining transactions.
        """
        time = 0.0
        while True:
            delay = self._config.inter_arrival.sample(self._rng)
            time += delay
            txn = self._create_transaction(time)
            yield delay, txn

    def _create_transaction(self, submit_time: float) -> Transaction:
        """Create transaction with sampled parameters."""
        self._txn_counter += 1

        # Sample runtime
        runtime = self._config.runtime.sample(self._rng)

        # Sample tables
        tables_read, tables_written = self._table_selector.select(
            self._config.tables_per_txn,
            self._config.num_tables,
            self._rng,
        )

        # Sample partitions (if enabled)
        partitions_written = None
        if self._config.partitions_per_txn is not None:
            partitions_written = {}
            for table_id in tables_written:
                n_parts = self._config.partitions_per_table[table_id]
                _, pw = self._partition_selector.select(
                    self._config.partitions_per_txn,
                    n_parts,
                    self._rng,
                )
                partitions_written[table_id] = pw

        # Sample operation type
        op_type = self._sample_operation_type()

        # Create appropriate transaction type
        common = dict(
            txn_id=self._txn_counter,
            submit_time_ms=submit_time,
            runtime_ms=runtime,
            tables_written=tables_written,
            partitions_written=partitions_written,
        )

        if op_type == 'fast_append':
            return FastAppendTransaction(**common)
        elif op_type == 'merge_append':
            return MergeAppendTransaction(
                **common,
                manifests_per_concurrent_commit=self._config.manifests_per_concurrent_commit,
            )
        else:
            return ValidatedOverwriteTransaction(**common)

    def _sample_operation_type(self) -> str:
        """Sample operation type from configured weights."""
        return str(self._rng.choice(
            ['fast_append', 'merge_append', 'validated_overwrite'],
            p=self._op_weights,
        ))
