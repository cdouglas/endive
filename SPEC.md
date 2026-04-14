# Endive Simulator Specification

**Version**: 3.1
**Date**: 2026-04-10

## Executive Summary

Endive is a discrete-event simulator for Apache Iceberg's optimistic concurrency control (OCC). It models catalog contention, conflict resolution, and commit latency under varying workloads across cloud storage providers. The architecture uses independent modules with clear API boundaries and encapsulated state.

### Design Principles

1. **Generator-Based I/O**: All latency-bearing operations yield bare `float` values (milliseconds). Only the `Simulation` runner converts these to SimPy timeouts.
2. **Encapsulation**: Internal state is private; modules interact only through defined APIs.
3. **Immutability**: Snapshots, configs, and results are frozen dataclasses.
4. **Opaque Distributions**: Latency sampling is always delegated to `LatencyDistribution` objects, never computed inline.
5. **Fixed Topology**: Tables and partitions are fixed at simulation start and owned by the Workload, not the Catalog.

---

## Module Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            Simulation                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                        SimPy Environment                       │  │
│  │            (_drive_generator: float → env.timeout)             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                  │                                   │
│        ┌─────────────────────────┼─────────────────────────┐        │
│        │                         │                         │        │
│        ▼                         ▼                         ▼        │
│  ┌───────────┐           ┌─────────────┐           ┌────────────┐  │
│  │ Workload  │──────────▶│ Transaction │──────────▶│  Catalog   │  │
│  │ Generator │           │  (active)   │           │(CAS/Append/│  │
│  └───────────┘           └─────────────┘           │  Instant)  │  │
│        │                        │                  └─────┬──────┘  │
│        │                        │                        │         │
│        │                        │                        ▼         │
│        │                        │                  ┌────────────┐  │
│        │                        └─────────────────▶│  Storage   │  │
│        │                                           │  Provider  │  │
│        │                                           └────────────┘  │
│        │                                                 │         │
│        ▼                                                 ▼         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                      Statistics Collector                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Module Layout

```
endive/
├── storage.py           # StorageProvider ABC, latency distributions, concrete providers
├── catalog.py           # Catalog ABC, CASCatalog, AppendCatalog, InstantCatalog
├── transaction.py       # Transaction ABC, FastAppend, ValidatedOverwrite
├── conflict_detector.py # Probabilistic and PartitionOverlap conflict detectors
├── workload.py          # Workload, WorkloadConfig, table/partition selectors
├── simulation.py        # Simulation runner, SimulationConfig, Statistics
├── config.py            # TOML loading, PROVIDER_PROFILES, validation
├── main.py              # CLI entry point, experiment directory management
├── saturation_analysis.py  # Analysis/plotting pipeline
├── test_utils.py        # create_test_config() helper
└── utils.py             # get_git_sha()
```

---

## 1. Storage Provider

The `StorageProvider` abstracts cloud object storage with latency-bearing operations. Latencies are drawn from opaque `LatencyDistribution` objects provided at construction. Every provider holds a seeded `np.random.RandomState` for determinism.

### 1.1 Interface

```python
@dataclass(frozen=True)
class StorageResult:
    success: bool
    latency_ms: float
    data_size_bytes: int

class LatencyDistribution(ABC):
    @abstractmethod
    def sample(self, rng: np.random.RandomState) -> float:
        """Draw a latency sample in milliseconds."""
        ...

class StorageProvider(ABC):
    def __init__(self, rng: np.random.RandomState): ...

    @abstractmethod
    def read(self, key: str, expected_size_bytes: int) -> Generator[float, None, StorageResult]: ...
    @abstractmethod
    def write(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]: ...
    @abstractmethod
    def cas(self, key: str, expected_version: int, size_bytes: int) -> Generator[float, None, StorageResult]: ...
    @abstractmethod
    def append(self, key: str, offset: int, size_bytes: int) -> Generator[float, None, StorageResult]: ...
    @abstractmethod
    def tail_append(self, key: str, size_bytes: int) -> Generator[float, None, StorageResult]: ...

    @property
    def supports_cas(self) -> bool: ...
    @property
    def supports_append(self) -> bool: ...
    @property
    def supports_tail_append(self) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def min_latency_ms(self) -> float: ...
```

Unsupported operations raise `UnsupportedOperationError`.

### 1.2 Latency Distributions

```python
@dataclass(frozen=True)
class LognormalLatency(LatencyDistribution):
    """Lognormal with minimum floor. YCSB-derived parameters."""
    mu: float           # ln(median)
    sigma: float        # tail heaviness
    min_latency_ms: float = 1.0

    @classmethod
    def from_median(cls, median_ms: float, sigma: float,
                    min_latency_ms: float = 1.0) -> LognormalLatency: ...

@dataclass(frozen=True)
class SizeBasedLatency(LatencyDistribution):
    """Size-dependent model: base + rate * size_mib + noise.
    Based on Durner et al. VLDB 2023 measurements."""
    base_latency_ms: float
    latency_per_mib_ms: float
    sigma: float
    min_latency_ms: float = 1.0

    def with_size(self, size_bytes: int) -> SizeBasedLatency: ...

@dataclass(frozen=True)
class FixedLatency(LatencyDistribution):
    """Deterministic latency for testing."""
    latency_ms: float
```

### 1.3 Concrete Providers

| Provider | `read` | `write` | `cas` | `append` | `tail_append` | min_latency_ms |
|----------|--------|---------|-------|----------|---------------|----------------|
| S3 Standard | yes | yes | yes | no | no | 43 |
| S3 Express | yes | yes | yes | yes | no | 10 |
| Azure Standard | yes | yes | yes | yes | no | 51 |
| Azure Premium | yes | yes | yes | yes | no | 40 |
| GCP | yes | yes | yes | no | no | 118 |
| Instant | yes | yes | yes | yes | yes | 1 |

Providers are constructed via `create_provider(name, rng, profiles)` using `PROVIDER_PROFILES` from `config.py`.

### 1.4 Provider Latency Profiles (YCSB June 2025)

| Provider | CAS median | CAS sigma | Append median | Append failure median | PUT model |
|----------|-----------|-----------|--------------|----------------------|-----------|
| S3 | 61ms | 0.14 | n/a | n/a | 30ms + 20ms/MiB |
| S3X | 22ms | 0.22 | 21ms | 23ms | 10ms + 10ms/MiB |
| Azure | 93ms | 0.82 | 87ms | 2072ms | 50ms + 25ms/MiB |
| AzureX | 64ms | 0.73 | 70ms | 2534ms | 30ms + 15ms/MiB |
| GCP | 170ms | 0.91 | n/a | n/a | 40ms + 17ms/MiB |
| Instant | 1ms | 0.1 | 1ms | 1ms | 0.5ms + 0.1ms/MiB |

---

## 2. Catalog

The `Catalog` manages optimistic concurrency control for table metadata. It exposes only `read()` and `commit()` to transactions. The commit mechanism (CAS vs append) is an internal implementation detail.

### 2.1 Core Types

```python
@dataclass(frozen=True)
class TableMetadata:
    table_id: int
    version: int
    num_partitions: int
    partition_versions: Tuple[int, ...]  # Per-partition version vector

    def with_version(self, new_version: int) -> TableMetadata: ...
    def with_partition_version(self, partition_id: int, new_version: int) -> TableMetadata: ...

@dataclass(frozen=True)
class CatalogSnapshot:
    """Immutable snapshot — the ONLY way transactions observe catalog state."""
    seq: int                            # Global sequence number (total ordering)
    tables: Tuple[TableMetadata, ...]
    timestamp_ms: float

    def get_table(self, table_id: int) -> TableMetadata: ...
    def get_partition_version(self, table_id: int, partition_id: int) -> int: ...

@dataclass(frozen=True)
class CommitResult:
    """Uniform result of Catalog.commit().
    On success: snapshot=None (transaction knows its writes were installed).
    On failure: snapshot=None (CAS/append do not return catalog content;
    transaction must call catalog.read() to learn the current state)."""
    success: bool
    latency_ms: float

@dataclass(frozen=True)
class IntentionRecord:
    """For append-based catalog commits with preconditions."""
    txn_id: int
    expected_seq: int
    tables_written: Dict[int, int]       # table_id -> new_version
    partitions_written: Dict[int, Tuple[int, ...]] | None = None
    size_bytes: int = 100
```

**Contention model note:** The global `seq` models a single-file catalog (`FileIOCatalog`) where all tables contend on one atomic pointer. Every commit—regardless of which table it targets—must increment the same `seq`, so concurrent writers to different tables still produce CAS failures. However, **cross-table CAS failures are cheap to retry**: the transaction reads the updated catalog, sees the intervening commit was to a different table, and retries the CAS without any manifest I/O. Only same-table conflicts with overlapping partitions require full conflict resolution. This distinction is critical for multi-table workloads: more tables means more CAS failures but cheaper retries, so the net effect depends on the balance between catalog round-trip cost and manifest I/O cost. A per-table metadata catalog (e.g., REST catalog backed by a database) would version each table independently, eliminating cross-table CAS failures entirely.

Internal types `_CASResult`, `_AppendResult`, and `_MutableTable` are not exposed to transactions.

### 2.2 Catalog Interface

```python
class Catalog(ABC):
    @abstractmethod
    def read(self, timestamp_ms: float = 0.0) -> Generator[float, None, CatalogSnapshot]: ...

    @abstractmethod
    def commit(
        self,
        expected_seq: int,
        writes: Dict[int, int],          # table_id -> new_version
        timestamp_ms: float = 0.0,
        intention: Optional[IntentionRecord] = None,
        partitions_written: Optional[Dict[int, FrozenSet[int]]] = None,
    ) -> Generator[float, None, CommitResult]: ...

    @property
    @abstractmethod
    def seq(self) -> int: ...
```

### 2.3 Implementations

**CASCatalog**: Single round-trip CAS on underlying storage. On success, applies writes atomically, advances partition versions for written partitions, and advances `seq` by 1. On failure, returns `CommitResult(success=False)` — the caller must call `catalog.read()` to learn the current state.

```python
class CASCatalog(Catalog):
    def __init__(self, storage: StorageProvider, num_tables: int,
                 partitions_per_table: Tuple[int, ...]): ...
```

**AppendCatalog**: Two internal round-trips (append + discovery read). The transaction sees only the final `CommitResult`, identical in shape to `CASCatalog`.

```python
class AppendCatalog(Catalog):
    def __init__(self, storage: StorageProvider, num_tables: int,
                 partitions_per_table: Tuple[int, ...]): ...
```

**InstantCatalog**: Fixed-latency CAS for testing. No `StorageProvider` required.

```python
class InstantCatalog(Catalog):
    def __init__(self, num_tables: int, partitions_per_table: Tuple[int, ...],
                 latency_ms: float = 1.0): ...
```

### 2.4 Commit Protocols

From the Transaction's perspective, the commit protocol is uniform: call `catalog.commit()` and receive a `CommitResult`.

```
Transaction                              Catalog
    │                                       │
    ├──── commit(seq, writes) ─────────────▶│
    │                                       │  [CAS or append+read internally]
    │◀──── CommitResult ───────────────────│
    │                                       │
    │  success=True:  done                  │
    │  success=False: must call read()      │
    │                                       │
    ├──── read() ──────────────────────────▶│  [on failure only]
    │◀──── CatalogSnapshot ────────────────│
    │                                       │
    │  [check write overlap, then retry     │
    │   or resolve conflict]                │
```

**CAS-based** (internal):
```
Catalog                                 Storage
    ├──── cas(key, expected_ver) ───────▶│
    │◀──── StorageResult ──────────────│
    │  [single round-trip]              │
```

**Append-based** (internal):
```
Catalog                                 Storage
    ├──── append(key, offset, data) ───▶│  (1. physical append)
    │◀──── StorageResult ──────────────│
    ├──── read(key) ───────────────────▶│  (2. discovery read)
    │◀──── StorageResult ──────────────│
    │  [two round-trips, hidden from Transaction]
```

---

## 3. Transaction Types

Transactions encapsulate the commit protocol and conflict handling. Each type has different conflict resolution behavior, but all use the same `catalog.commit()` interface.

### 3.1 Iceberg Metadata Model

A catalog commit involves three levels of metadata:

1. **Catalog file**: Global pointer to the current table metadata. Updated via CAS.
2. **Table metadata file**: Per-table state including partition versions and manifest list pointers. One file per table, stored in object storage.
3. **Manifest list**: Per-partition list of manifest files. Stored in object storage.

When `metadata_inlined=True`, the table metadata is stored inside the catalog CAS object. The table metadata file no longer exists as a separate object, eliminating its read/write cost. The manifest list is NOT inlined — it remains a separate storage object.

Each partition maintains its own entry in the catalog. Disjoint partitions do not interact: a CAS failure caused by a commit to a different partition is a free retry (catalog re-read + CAS only).

### 3.2 Per-Attempt I/O Cost Model

Every commit attempt pays storage I/O before the CAS. The cost depends on whether table metadata is inlined and how many partitions are written.

**Non-inlined (`metadata_inlined=False`, exp1–5):**

| Step | Operation | Count | Key |
|------|-----------|-------|-----|
| 1 | Read table metadata | 1 | `table_metadata` |
| 2 | Read manifest list | N | `manifest_list` |
| 3 | Write manifest list | N | `manifest_list` |
| 4 | Write table metadata | 1 | `table_metadata` |

N = number of partitions written (first attempt) or overlapping partitions (retry).

**Inlined (`metadata_inlined=True`, exp6):**

| Step | Operation | Count | Key |
|------|-----------|-------|-----|
| 1 | Read manifest list | N | `manifest_list` |
| 2 | Write manifest list | N | `manifest_list` |

Table metadata reads/writes are eliminated — the state is in the catalog CAS object.

**In ML+ mode** (`ml_append_mode=True`): ML writes are eliminated (the append replaces the rewrite). TM reads/writes are still paid (if non-inlined).

```python
def get_per_attempt_cost(self, ml_append_mode, n_partitions=1, metadata_inlined=False):
    return ConflictCost(
        table_metadata_reads=0 if metadata_inlined else 1,
        table_metadata_writes=0 if metadata_inlined else 1,
        manifest_list_reads=n_partitions,
        manifest_list_writes=0 if ml_append_mode else n_partitions,
    )
```

### 3.3 Commit Protocol — Full Pseudocode

```python
def execute(catalog, storage, conflict_detector, max_retries, ml_append_mode, metadata_inlined):
    # Phase 1: Read catalog snapshot
    start_snapshot = yield from catalog.read()

    # Phase 2: Execute transaction work
    yield runtime_ms

    # Phase 3: Commit loop
    last_snapshot = start_snapshot
    per_attempt_n = count(partitions_written)        # All written partitions on first attempt

    for attempt in range(max_retries + 1):

        # --- Per-attempt I/O (skipped on disjoint retry where per_attempt_n=0) ---
        if per_attempt_n > 0:
            cost = get_per_attempt_cost(ml_append_mode, per_attempt_n, metadata_inlined)
            yield from pay_io(cost, storage)         # TM read, ML reads, ML writes, TM write

        # --- CAS ---
        result = yield from catalog.commit(last_snapshot.seq, writes)
        if result.success:
            return COMMITTED

        # --- CAS failed: read catalog to determine overlap ---
        current_snapshot = yield from catalog.read()

        # Non-inlined: separate TM read needed to get partition versions
        if not metadata_inlined:
            yield from storage.read("table_metadata")

        overlap = compute_write_overlap(last_snapshot, current_snapshot)

        # --- Conflict resolution (only when partitions overlap) ---
        if overlap.has_overlap:
            conflict_cost = get_conflict_cost(n_table_versions_behind, ...)
            yield from pay_io(conflict_cost, storage)  # VO: historical ML reads

            if can_have_real_conflict():
                if conflict_detector.is_real_conflict(...) and should_abort():
                    return ABORTED("validation_exception")

        # --- Prepare for retry ---
        last_snapshot = current_snapshot
        per_attempt_n = overlap.n_partitions          # 0 = free retry next iteration

    return ABORTED("max_retries_exceeded")
```

### 3.4 I/O Cost Summary Table

| Path | Non-inlined storage ops | Inlined storage ops |
|------|------------------------|---------------------|
| **First attempt** | TM_r(1) + ML_r(N) + ML_w(N) + TM_w(1) + CAS | ML_r(N) + ML_w(N) + CAS |
| **Failure path** | catalog_read + TM_r(1) → overlap check | catalog_read → overlap check |
| **Retry, disjoint** | CAS only (free) | CAS only (free) |
| **Retry, overlap (FA)** | TM_r(1) + ML_r(M) + ML_w(M) + TM_w(1) + CAS | ML_r(M) + ML_w(M) + CAS |
| **Retry, overlap (VO)** | above + (V-1)×M historical ML reads | above + (V-1)×M historical ML reads |

N = partitions written, M = overlapping partitions, V = per-table version delta.

### 3.5 Core Types

```python
class TransactionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTED = auto()

@dataclass(frozen=True)
class ConflictCost:
    """I/O operations for one phase of the commit protocol."""
    table_metadata_reads: int = 0   # TM file reads (non-inlined only)
    table_metadata_writes: int = 0  # TM file writes (non-inlined only)
    manifest_list_reads: int = 0    # ML reads (per partition)
    manifest_list_writes: int = 0   # ML writes (per partition)
    historical_ml_reads: int = 0    # I/O convoy for validation (VO only)

@dataclass(frozen=True)
class TransactionResult:
    status: TransactionStatus
    txn_id: int
    commit_time_ms: float              # -1 if not committed
    abort_time_ms: float               # -1 if not aborted
    abort_reason: Optional[str]
    total_retries: int
    commit_latency_ms: float           # Time in commit protocol
    total_latency_ms: float            # End-to-end time
    operation_type: str                # "fast_append", "validated_overwrite"
    runtime_ms: float

    # I/O tracking
    table_metadata_reads: int
    table_metadata_writes: int
    manifest_list_reads: int
    manifest_list_writes: int

    # Timing decomposition (ms)
    catalog_read_ms: float             # Catalog reads + failure-path TM reads
    per_attempt_io_ms: float           # Total time in per-attempt storage I/O
    conflict_io_ms: float              # Total time in retry-specific I/O (convoy)
    catalog_commit_ms: float           # Total time in catalog.commit() calls

    # Retry characterization
    catalog_conflicts: int = 0         # CAS failures with no write overlap (free retries)
    tblptn_conflicts: int = 0          # CAS failures with write overlap
    max_snapshots_behind: int = 0      # Max (new_seq - old_seq) across retries
```

### 3.6 Transaction ABC

```python
class Transaction(ABC):
    def __init__(self, txn_id, submit_time_ms, runtime_ms,
                 tables_written: FrozenSet[int],
                 partitions_written: Dict[int, FrozenSet[int]]): ...

    def execute(self, catalog, storage, conflict_detector,
                max_retries=10, ml_append_mode=False,
                metadata_inlined=False) -> Generator[float, None, TransactionResult]: ...

    def get_per_attempt_cost(self, ml_append_mode, n_partitions=1,
                             metadata_inlined=False) -> ConflictCost: ...

    @abstractmethod
    def get_conflict_cost(self, n_snapshots_behind, ml_append_mode,
                          n_partitions=1) -> ConflictCost: ...

    def compute_write_overlap(self, old_snapshot, new_snapshot) -> WriteOverlap: ...
```

### 3.7 Write Overlap Detection

```python
def compute_write_overlap(self, old_snapshot, new_snapshot) -> WriteOverlap:
    overlapping = {}
    for table_id in self.tables_written:
        old_table = old_snapshot.get_table(table_id)
        new_table = new_snapshot.get_table(table_id)
        if old_table.version == new_table.version:
            continue  # Table not modified by intervening commits
        overlap_pids = frozenset(
            pid for pid in self.partitions_written.get(table_id, ())
            if old_table.partition_versions[pid] != new_table.partition_versions[pid]
        )
        if overlap_pids:
            overlapping[table_id] = overlap_pids
    return WriteOverlap(overlapping) if overlapping else NO_OVERLAP
```

### 3.8 Concrete Transaction Types

**FastAppendTransaction**: Append-only, no validation, no real conflicts possible. Always retries. No additional I/O beyond per-attempt cost.

```python
class FastAppendTransaction(Transaction):
    operation_type = "fast_append"
    can_have_real_conflict() -> False
    get_conflict_cost(...) -> ConflictCost()  # No additional retry cost
```

**ValidatedOverwriteTransaction**: Full validation. Can have real conflicts. Aborts on real conflict. Additional retry cost: historical ML reads (I/O convoy).

```python
class ValidatedOverwriteTransaction(Transaction):
    operation_type = "validated_overwrite"
    can_have_real_conflict() -> True
    should_abort_on_real_conflict() -> True
    get_conflict_cost(n_table_versions_behind, ml_append_mode, n_partitions):
        # Read historical MLs at versions K+1..K+N-1 to validate read set.
        # The per-attempt cost reads the ML at version K+N, so N-1 additional.
        return ConflictCost(
            historical_ml_reads=max(0, n_table_versions_behind - 1) * n_partitions,
        )
```

`n_table_versions_behind` is the per-table version delta, NOT the catalog sequence delta. The convoy is computed per-table in `_commit_loop`, so each table's version delta is paired with that table's overlapping partition count. For multi-table VO transactions, this correctly yields `(V_A-1)×M_A + (V_B-1)×M_B` instead of overcounting with `(V_A+V_B-1)×(M_A+M_B)`.

### 3.9 ML+ Manifest List Protocol

In ML+ mode (`ml_append_mode=True`), ML writes are eliminated from the per-attempt cost (the append replaces the rewrite). TM reads/writes are still paid if non-inlined.

### 3.10 Inlined Table Metadata

When `metadata_inlined=True`:

1. Table metadata is stored in the catalog CAS object — no separate file.
2. Per-attempt cost eliminates TM read and TM write (2 fewer ops per attempt).
3. Failure-path catalog read already contains partition versions — no extra TM read.
4. CAS payload grows with each commit (`commit_growth_bytes`), capped at `max_catalog_size_bytes`. CAS latency scales with payload size via `SizeBasedLatency`.
5. The VO I/O convoy is NOT eliminated — historical MLs must still be read to validate the read set.

---

## 4. Conflict Detection

Conflict detection determines whether a catalog conflict involves real data overlap or is a false conflict between unrelated changes.

```python
class ConflictDetector(ABC):
    """Defined in transaction.py alongside Transaction (it's part of the commit protocol)."""
    @abstractmethod
    def is_real_conflict(self, txn: Transaction, current_snapshot: CatalogSnapshot,
                         start_snapshot: CatalogSnapshot) -> bool: ...
```

### Implementations (in `conflict_detector.py`)

**ProbabilisticConflictDetector**: Returns real conflict with configured probability. Respects `txn.can_have_real_conflict()` (FastAppend always returns False). Uses seeded RNG for determinism.

```python
class ProbabilisticConflictDetector(ConflictDetector):
    def __init__(self, real_conflict_probability: float,
                 rng: np.random.RandomState | None = None): ...
```

**PartitionOverlapConflictDetector**: Checks per-(table, partition) version changes between start and current snapshots. Real conflict if any written partition was modified by a concurrent transaction.

```python
class PartitionOverlapConflictDetector(ConflictDetector):
    def is_real_conflict(self, txn, current, start) -> bool: ...
```

---

## 5. Workload Generator

The `Workload` generates transactions with encapsulated rate and parameters. Topology (tables, partitions) is owned by the Workload, not the Catalog.

### 5.1 Selectors

```python
class TableSelector(ABC):
    @abstractmethod
    def select(self, n_tables: int, total_tables: int,
               rng: np.random.RandomState) -> Tuple[FrozenSet[int], FrozenSet[int]]:
        """Returns (tables_read, tables_written) where tables_written ⊆ tables_read."""

class UniformTableSelector(TableSelector): ...
class ZipfTableSelector(TableSelector):
    def __init__(self, alpha: float = 1.5, write_fraction: float = 1.0): ...

class PartitionSelector(ABC):
    @abstractmethod
    def select(self, n_partitions: int, total_partitions: int,
               rng: np.random.RandomState) -> Tuple[FrozenSet[int], FrozenSet[int]]: ...

class UniformPartitionSelector(PartitionSelector): ...
class ZipfPartitionSelector(PartitionSelector):
    def __init__(self, alpha: float = 1.5, write_fraction: float = 1.0): ...
```

Zipf PMF: `P(k) = (1/k^alpha) / sum(1/i^alpha for i in 1..n)`

### 5.2 WorkloadConfig

```python
@dataclass(frozen=True)
class WorkloadConfig:
    inter_arrival: LatencyDistribution
    runtime: LatencyDistribution
    num_tables: int
    partitions_per_table: Tuple[int, ...]

    fast_append_weight: float = 0.7
    validated_overwrite_weight: float = 0.3

    tables_per_txn: int = 1
    table_selector: Optional[TableSelector] = None       # None = uniform
    partitions_per_txn: Optional[int] = None
    partition_selector: Optional[PartitionSelector] = None  # None = uniform
```

### 5.3 Workload

```python
class Workload:
    def __init__(self, config: WorkloadConfig, seed: Optional[int] = None): ...

    def generate(self) -> Generator[Tuple[float, Transaction], None, None]:
        """Yield (inter_arrival_delay_ms, Transaction) pairs indefinitely."""
```

The `generate()` method samples inter-arrival times, runtime, operation type, table/partition selections, and constructs the appropriate `Transaction` subclass. Operation type weights are normalized internally.

When `partitions_per_txn` is None (the default), the Workload generates `{tid: frozenset({0})}` for each written table — modeling unpartitioned tables as single-partition. This ensures `partitions_written` is always a populated `Dict[int, FrozenSet[int]]`, enabling partition-level overlap detection in the commit loop without Optional paths.

---

## 6. Simulation Runner

The `Simulation` class is the only place SimPy is used. All other components yield bare floats.

### 6.1 SimulationConfig

```python
@dataclass(frozen=True)
class SimulationConfig:
    duration_ms: float
    seed: Optional[int]

    storage_provider: StorageProvider
    catalog: Catalog
    workload: Workload
    conflict_detector: ConflictDetector

    max_retries: int = 10
    ml_append_mode: bool = False
    metadata_inlined: bool = False     # Passed to txn.execute()
```

### 6.2 Statistics

```python
@dataclass
class Statistics:
    transactions: List[TransactionResult]

    # Aggregate counters
    committed: int
    aborted: int
    total_retries: int
    validation_exceptions: int
    table_metadata_reads: int
    table_metadata_writes: int
    manifest_list_reads: int
    manifest_list_writes: int

    def record_transaction(self, result: TransactionResult) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def export_parquet(self, path: str) -> None: ...
```

Output DataFrame columns: `txn_id`, `t_submit`, `t_runtime`, `t_commit`, `commit_latency`, `total_latency`, `n_retries`, `status`, `operation_type`, `abort_reason`, `table_metadata_reads`, `table_metadata_writes`, `manifest_list_reads`, `manifest_list_writes`, `catalog_read_ms`, `per_attempt_io_ms`, `conflict_io_ms`, `catalog_commit_ms`, `event_count`.

For streaming export (lower memory), pass `output_path` to the `Statistics` constructor or to `Simulation`. Results are written incrementally to parquet in batches, avoiding accumulation in memory.

### 6.3 `_CountingEnvironment`

```python
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
```

Used instead of `simpy.Environment` to instrument DES engine performance. The `event_count` and `queue_depth` are sampled periodically for profiling output.

### 6.4 Simulation

```python
class Simulation:
    def __init__(self, config: SimulationConfig,
                 output_path: str | None = None,
                 progress_path: str | None = None,
                 profile: bool = False): ...
    def run(self) -> Statistics: ...
```

The runner:
1. Seeds `np.random` from `config.seed`
2. Creates a `_CountingEnvironment`
3. Iterates `workload.generate()`, yielding `env.timeout(delay)` for each inter-arrival
4. Launches each transaction as a SimPy process
5. Uses `_drive_generator(env, gen)` to bridge latency-yielding generators with `env.timeout()`
6. Records each `TransactionResult` in `Statistics`
7. Runs a progress reporter that writes `.progress.json` with DES rate, queue depth, and simulation speed
8. If `profile=True`, collects periodic samples and writes `.profile.json` at completion
9. Runs until `duration_ms`

### 6.5 DES Engine Profiling

When `profile=True`, the simulation samples engine metrics every 60 simulated seconds and writes `.profile.json` alongside the results:

```json
{
    "summary": {
        "des_events_total": 210000,
        "des_rate_mean": 65000.0,
        "des_rate_min": 55000.0,
        "des_rate_max": 72000.0,
        "queue_depth_max": 4200,
        "queue_depth_mean": 2100.0,
        "peak_processes": 4068,
        "wall_clock_seconds": 3.2,
        "sim_speed_min": 800.0
    },
    "samples": [
        {"sim_time_ms": 60000, "wall_clock_s": 0.08, ...},
        ...
    ]
}
```

The progress reporter always includes `des_rate` and `queue_depth` in `.progress.json` (backward-compatible). The `event_count` per transaction is written to the parquet output and tracks how many SimPy events each transaction consumed.

---

## 7. Configuration

Configuration is loaded from TOML and fully validated before constructing a `SimulationConfig`.

### 7.1 Entry Point

```python
def load_simulation_config(config_path: str, *, seed_override: int | None = None) -> SimulationConfig:
    """Load and validate TOML config. The ONLY entry point for configuration."""
```

Internally builds: storage provider, catalog, workload, and conflict detector. Topology (num_tables, partitions_per_table) is shared between catalog and workload at construction time.

### 7.2 TOML Schema

```toml
[simulation]
duration_ms = 3600000
seed = 42                              # Optional; overridable via seed_override
output_path = "results.parquet"

[experiment]
label = "exp_baseline"                 # Optional; enables directory structure

[storage]
provider = "s3x"                       # s3, s3x, azure, azurex, gcp, instant

[catalog]
num_tables = 1
num_groups = 1
table_metadata_inlined = false         # See §3.10
# Inlined-only knobs (exp6):
# initial_partition_size_bytes = 16000
# commit_growth_bytes = 0

[catalog.partition]
num_partitions = 32                    # Uniform across tables
# per_table = [10, 20, 30]             # Or explicit per-table

[transaction]
retry = 10
runtime.min = 30000
runtime.mean = 180000
runtime.sigma = 1.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
real_conflict_probability = 0.0
manifest_list_mode = "rewrite"         # "rewrite" or "append" (ML+ mode)
operation_types.fast_append = 0.7
operation_types.validated_overwrite = 0.3

[partition]                            # Optional: enables PartitionOverlapConflictDetector
enabled = true
partitions_per_txn = 1                 # How many partitions each txn writes
# selection.distribution = "zipf"      # Optional: "uniform" (default) or "zipf"
# selection.zipf_alpha = 1.5
```

Experiment configs may also include a `[plots]` section consumed by `scripts/regenerate_plots.py`; it is ignored by the simulator.

### 7.3 Experiment Hash

`compute_experiment_hash()` creates a deterministic hash from config parameters (excludes seed, output_path, experiment.label). Same parameters with the same code produce the same hash and share a directory.

### 7.3.1 Template Provenance

Each generated variant carries template provenance stamps under `[experiment]`:

- `template_path` — relative path to the source template in `experiment_configs/`.
- `template_hash` — hash of the template itself via `compute_template_hash()`, which mirrors `compute_experiment_hash` but strips `[experiment]` (including these stamps), `[plots]`, and `simulation.seed`. Does **not** mix in the code hash — template drift is tracked separately from code drift.
- `template_overrides` — inline table of sweep parameters applied to produce this variant.

`expctl list` surfaces staleness reasons independently:

- `code` — `version.txt`'s `code_hash` differs from the current simulator code hash.
- `self-hash` — stored `cfg.toml` no longer hashes to the dir name under the current code (corruption or manual edit after the run). Suppressed when `code` drift is already flagged, because the digest would naturally differ.
- `template` — stored `template_hash` differs from `compute_template_hash()` of the live `experiment_configs/<label>.toml`. Catches silent template edits (e.g. flipping `table_metadata_inlined`).
- `template-missing` — source template was deleted or renamed.

Multiple reasons can apply to a single dir; the list status shows them comma-separated: `stale (code, template)`.

```
experiments/
├── exp_baseline-a3f7b2/
│   ├── cfg.toml                       # Configuration snapshot
│   ├── version.txt                    # Git SHA
│   ├── 42/results.parquet             # Seed 42 results
│   └── 43/results.parquet             # Seed 43 results
└── consolidated.parquet               # All experiments merged
```

---

## 8. Invariants

### 8.1 Version Monotonicity
- `Catalog.seq` advances by exactly 1 on each successful commit
- Never decreases or skips values

### 8.2 Snapshot Isolation
- Transactions observe catalog state via immutable `CatalogSnapshot`
- Changes are only visible after commit

### 8.3 I/O Convoy Exactness (ValidatedOverwrite only)
- When N table versions behind AND write overlap exists, read exactly (N-1) × M historical manifest lists
- N = per-table version delta for overlapping tables; M = number of overlapping partitions
- The per-attempt cost reads the ML at version K+N; only K+1..K+N-1 need additional reads
- Skipped entirely when there is no write overlap (cross-table or disjoint partitions)

### 8.4 Conflict Type Distinction
- **No overlap**: Different table or disjoint partitions — no manifest I/O, just re-CAS
- **False conflict**: Same table + overlapping partitions, but no data conflict — merge and retry
- **Real conflict**: Same table + overlapping partitions with data conflict — may abort (operation-dependent)

### 8.5 Determinism
- Same seed + same config produces identical results
- All randomness uses seeded `np.random.RandomState`

### 8.6 Minimum Latency
- All operations have a minimum latency floor (provider-specific)
- Prevents unrealistic zero-latency scenarios

### 8.7 Uniform Catalog Interface
- Transactions call only `read()` and `commit()` on the Catalog
- Transactions do not know whether the underlying mechanism is CAS or append
- `commit()` returns `CommitResult` with the same semantics for all implementations:
  - On success: transaction knows its state was installed
  - On failure: transaction must call `catalog.read()` to learn the current state

### 8.8 Commit Does Not Return State
- Neither success nor failure returns a catalog snapshot
- On success: the transaction knows its state was installed (CAS/append guarantees atomicity)
- On failure: the transaction must call `catalog.read()` to get the current snapshot, paying one read round-trip
- This models the real cost: CAS returns only success/failure, not the current value

### 8.9 Information Asymmetry in Append Protocol
- `Storage.append()` returns only physical success (offset matched)
- Logical outcome (preconditions satisfied) is never returned to the caller
- The Catalog performs a discovery read to determine the outcome internally
- This complexity is hidden by the uniform `commit()` interface

### 8.10 ML+ Deferred Validity
- In ML+ mode, manifest list appends are tentative until catalog commit
- The ML entry's validity is determined by the catalog commit outcome
- Transaction must read an ML containing all committed transactions before retry

### 8.11 Topology Ownership
- Table and partition counts are fixed at simulation start
- The Workload owns topology and configures Transactions accordingly
- The Catalog does not expose topology queries

### 8.12 Cross-Table / Disjoint Retry Cost
- CAS failures caused by commits to a different table (or disjoint partitions of the same table) do not require manifest or table metadata I/O
- Each partition has its own entry in the catalog; disjoint writes do not interact
- The retry cost is: 1 catalog read (+ 1 TM read if non-inlined) + 1 CAS round-trip
- Per-attempt I/O and conflict resolution costs apply ONLY when `compute_write_overlap()` returns overlap
- Critical for multi-table workloads: with N tables and uniform selection, ~(N-1)/N of CAS failures are cross-table and essentially free to retry

### 8.13 Table Metadata Inlining
- When `metadata_inlined=True`, table metadata is stored in the catalog CAS object
- Per-attempt cost eliminates TM read and TM write (2 fewer storage ops per attempt)
- Failure-path catalog read includes partition versions (no extra TM read)
- CAS payload grows by `commit_growth_bytes` per successful commit, capped at `max_catalog_size_bytes`
- CAS latency scales with payload size via `SizeBasedLatency`

---

## Appendix A: Glossary

- **CAS**: Compare-and-swap; atomic conditional update
- **ML+**: Manifest list append mode; avoids ML rewrite on false conflict
- **Write Overlap**: Intervening commit modified the same table AND overlapping partitions as the retrying transaction
- **No Overlap (cross-table)**: Intervening commit was to a different table; retry is free (catalog read + CAS only)
- **No Overlap (disjoint partitions)**: Intervening commit was to the same table but different partitions; retry is free
- **False Conflict**: Same table + overlapping partitions, but no data conflict; merge and retry with manifest I/O
- **Real Conflict**: Same table + overlapping partitions with data conflict; may abort (operation-dependent)
- **I/O Convoy**: Reading N historical manifest lists for N missed snapshots (only when write overlap exists)
- **Snapshot Isolation**: Transaction sees consistent point-in-time state
- **Validation Exception**: Abort due to real data overlap detection
- **Table metadata (TM)**: Per-table metadata file stored in object storage (non-inlined) or in the catalog CAS object (inlined)
- **Manifest list (ML)**: Per-partition list of manifest files, stored in object storage
- **Per-attempt cost**: I/O paid on first attempt and on retries with write overlap: TM read + ML reads + ML writes + TM write (non-inlined) or ML reads + ML writes (inlined)
- **Conflict cost**: Additional I/O paid only on retry with write overlap (type-dependent: zero for FA, historical ML reads for VO)
