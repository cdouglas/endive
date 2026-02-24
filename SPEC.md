# Endive Simulator Specification

**Version**: 2.0
**Date**: 2026-02-23

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
├── transaction.py       # Transaction ABC, FastAppend, MergeAppend, ValidatedOverwrite
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
    ) -> Generator[float, None, CommitResult]: ...

    @property
    @abstractmethod
    def seq(self) -> int: ...
```

### 2.3 Implementations

**CASCatalog**: Single round-trip CAS on underlying storage. On CAS success, applies writes atomically and advances `seq` by 1. On failure, captures a snapshot for conflict resolution.

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

### 3.1 Core Types

```python
class TransactionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTED = auto()

@dataclass(frozen=True)
class ConflictCost:
    """I/O operations required to resolve a conflict."""
    metadata_reads: int = 0
    manifest_list_reads: int = 0
    manifest_list_writes: int = 0
    historical_ml_reads: int = 0   # I/O convoy for validation
    manifest_file_reads: int = 0
    manifest_file_writes: int = 0

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
    operation_type: str                # "fast_append", "merge_append", "validated_overwrite"
    runtime_ms: float

    # I/O tracking
    manifest_list_reads: int
    manifest_list_writes: int
    manifest_file_reads: int
    manifest_file_writes: int

    # Timing decomposition (ms)
    catalog_read_ms: float
    per_attempt_io_ms: float           # Total time in per-attempt storage I/O
    conflict_io_ms: float              # Total time in retry-specific I/O
    catalog_commit_ms: float           # Total time in catalog.commit() calls
```

### 3.2 Transaction ABC

```python
class Transaction(ABC):
    def __init__(self, txn_id: int, submit_time_ms: float, runtime_ms: float,
                 tables_written: FrozenSet[int],
                 partitions_written: Optional[Dict[int, FrozenSet[int]]] = None): ...

    def execute(self, catalog: Catalog, storage: StorageProvider,
                conflict_detector: ConflictDetector, max_retries: int = 10,
                ml_append_mode: bool = False) -> Generator[float, None, TransactionResult]: ...

    @abstractmethod
    def can_have_real_conflict(self) -> bool: ...
    @abstractmethod
    def should_abort_on_real_conflict(self) -> bool: ...

    def get_per_attempt_cost(self, ml_append_mode: bool) -> ConflictCost:
        """I/O cost paid on first attempt and on retries with write overlap.
        Cost: 1 ML read + 1 MF write + 1 ML write (0 ML write in ML+ mode).
        Skipped on retry when intervening commits have no write overlap."""

    @abstractmethod
    def get_conflict_cost(self, n_snapshots_behind: int, ml_append_mode: bool) -> ConflictCost:
        """Additional retry-specific I/O cost. Only paid when write overlap exists."""

    def has_write_overlap(self, old_snapshot: CatalogSnapshot,
                          new_snapshot: CatalogSnapshot) -> bool:
        """Check if intervening commits overlap with this transaction's writes.
        Returns False if all intervening commits were to different tables or
        disjoint partitions of the same table. Returns True if any intervening
        commit modified the same table AND overlapping partitions (or if
        partition tracking is disabled, any same-table modification)."""
```

The `execute()` method drives the transaction lifecycle:

1. **Read** catalog snapshot (`catalog.read()`)
2. **Execute** transaction work (yield `runtime_ms`)
3. **Commit loop** (up to `max_retries + 1` attempts):
   - a. Pay per-attempt I/O cost (ML read, MF write, ML write) — **skipped when no write overlap** (see below)
   - b. Call `catalog.commit()`
   - c. On success: return COMMITTED
   - d. On failure: read catalog (`catalog.read()`) to learn current state
   - e. Check **write overlap** (`has_write_overlap()`): did any intervening commit modify the same table AND overlapping partitions?
   - f. If **no overlap** (cross-table or disjoint partitions): skip to step 3b on next iteration (no manifest I/O)
   - g. If **overlap**: pay type-specific conflict cost, check for real conflict (may abort), go to step 3a on next iteration

The `_yield_from()` helper tracks elapsed time across sub-generators. The `_pay_conflict_cost()` method executes storage operations for each `ConflictCost` field.

### 3.3 Write Overlap Check

After a CAS failure, the transaction compares its `tables_written` and `partitions_written` against what changed between the old and new catalog snapshots:

```python
def has_write_overlap(self, old_snapshot, new_snapshot) -> bool:
    for table_id in self.tables_written:
        old_table = old_snapshot.get_table(table_id)
        new_table = new_snapshot.get_table(table_id)
        if old_table.version == new_table.version:
            continue  # This table was not modified by intervening commits
        # Same table was modified — check partition overlap
        if self.partitions_written is None:
            return True  # No partition tracking; assume overlap
        for pid in self.partitions_written.get(table_id, ()):
            if old_table.partition_versions[pid] != new_table.partition_versions[pid]:
                return True  # Overlapping partition
    return False
```

**No overlap** means: every intervening commit was either to a different table entirely, or to the same table but disjoint partitions. The transaction's manifest file and manifest list from the previous attempt are still valid — it just needs to retry the CAS with the updated seq.

**Overlap** means: at least one intervening commit modified the same table AND the same partition(s). The transaction must redo its manifest work (re-read ML, re-write MF, re-write ML) and perform type-specific conflict resolution.

### 3.4 Concrete Transaction Types

All per-attempt I/O costs and conflict costs below are paid **only when `has_write_overlap()` returns True**. Cross-table and disjoint-partition retries pay only the catalog read + CAS round-trip.

**FastAppendTransaction**: Append-only, no validation, no real conflicts possible. Always retries on conflict. No additional retry cost beyond per-attempt I/O.

```python
class FastAppendTransaction(Transaction):
    operation_type = "fast_append"
    can_have_real_conflict() -> False
    should_abort_on_real_conflict() -> False
    get_conflict_cost() -> ConflictCost()  # No additional retry cost
```

**MergeAppendTransaction**: Must re-merge manifests on conflict. No real conflicts. Always retries. Additional retry cost (with overlap): N manifest file reads + N writes, where N = `n_behind * manifests_per_concurrent_commit`.

```python
class MergeAppendTransaction(Transaction):
    operation_type = "merge_append"
    def __init__(self, *args, manifests_per_concurrent_commit: float = 1.5, **kwargs): ...
    can_have_real_conflict() -> False
    should_abort_on_real_conflict() -> False
    get_conflict_cost(n_behind, ml_append_mode) -> ConflictCost(
        manifest_file_reads=n_behind * manifests_per_commit,
        manifest_file_writes=n_behind * manifests_per_commit,
    )
```

**ValidatedOverwriteTransaction**: Full validation via `validationHistory()`. Can have real conflicts (data overlap). Aborts with `"validation_exception"` on real conflict. Additional retry cost (with overlap): N historical ML reads (I/O convoy).

```python
class ValidatedOverwriteTransaction(Transaction):
    operation_type = "validated_overwrite"
    can_have_real_conflict() -> True
    should_abort_on_real_conflict() -> True
    get_conflict_cost(n_behind, ml_append_mode) -> ConflictCost(
        historical_ml_reads=n_behind,  # I/O convoy
    )
```

### 3.5 ML+ Manifest List Protocol

In ML+ mode (`ml_append_mode=True`), the manifest list is updated via append before the catalog commit. This is Transaction-level logic:

1. Transaction appends ML entry (tentative)
2. Transaction calls `catalog.commit()` (uniform interface)
3. If committed: ML entry is valid
4. If conflict: ML entry is orphaned (harmless); transaction must read ML containing all committed transactions before retry

The per-attempt cost saves 1 ML write in ML+ mode (the append replaces the rewrite).

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

**ProbabilisticConflictDetector**: Returns real conflict with configured probability. Respects `txn.can_have_real_conflict()` (FastAppend and MergeAppend always return False). Uses seeded RNG for determinism.

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
    merge_append_weight: float = 0.2
    validated_overwrite_weight: float = 0.1

    tables_per_txn: int = 1
    table_selector: Optional[TableSelector] = None       # None = uniform
    partitions_per_txn: Optional[int] = None
    partition_selector: Optional[PartitionSelector] = None  # None = uniform

    manifests_per_concurrent_commit: float = 1.5
```

### 5.3 Workload

```python
class Workload:
    def __init__(self, config: WorkloadConfig, seed: Optional[int] = None): ...

    def generate(self) -> Generator[Tuple[float, Transaction], None, None]:
        """Yield (inter_arrival_delay_ms, Transaction) pairs indefinitely."""
```

The `generate()` method samples inter-arrival times, runtime, operation type, table/partition selections, and constructs the appropriate `Transaction` subclass. Operation type weights are normalized internally.

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

    backoff_enabled: bool = False
    backoff_base_ms: float = 10.0
    backoff_multiplier: float = 2.0
    backoff_max_ms: float = 5000.0
    backoff_jitter: float = 0.1
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
    manifest_list_reads: int
    manifest_list_writes: int
    manifest_file_reads: int
    manifest_file_writes: int

    def record_transaction(self, result: TransactionResult) -> None: ...
    def to_dataframe(self) -> pd.DataFrame: ...
    def export_parquet(self, path: str) -> None: ...
```

Output DataFrame columns: `txn_id`, `t_submit`, `t_runtime`, `t_commit`, `commit_latency`, `total_latency`, `n_retries`, `status`, `operation_type`, `abort_reason`, `manifest_list_reads`, `manifest_list_writes`, `manifest_file_reads`, `manifest_file_writes`, `catalog_read_ms`, `per_attempt_io_ms`, `conflict_io_ms`, `catalog_commit_ms`.

### 6.3 Simulation

```python
class Simulation:
    def __init__(self, config: SimulationConfig): ...
    def run(self) -> Statistics: ...
```

The runner:
1. Seeds `np.random` from `config.seed`
2. Creates a `simpy.Environment`
3. Iterates `workload.generate()`, yielding `env.timeout(delay)` for each inter-arrival
4. Launches each transaction as a SimPy process
5. Uses `_drive_generator(env, gen)` to bridge latency-yielding generators with `env.timeout()`
6. Records each `TransactionResult` in `Statistics`
7. Runs until `duration_ms`

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
# type = "cas"                         # Inferred from provider capabilities

[catalog.partitions]
num_partitions = 100                   # Uniform across tables
# per_table = [10, 20, 30]            # Or explicit per-table

[transaction]
retry = 10
runtime.mean = 180000
runtime.sigma = 1.5
inter_arrival.distribution = "exponential"
inter_arrival.scale = 100.0
real_conflict_probability = 0.0
manifest_list_mode = "rewrite"         # "rewrite" or "append" (ML+ mode)

[transaction.operation_types]
fast_append = 0.7
merge_append = 0.2
validated_overwrite = 0.1

[transaction.retry_backoff]
enabled = true
base_ms = 10.0
multiplier = 2.0
max_ms = 5000.0
jitter = 0.1
```

### 7.3 Experiment Hash

`compute_experiment_hash()` creates a deterministic hash from config parameters (excludes seed, output_path, experiment.label). Same parameters with the same code produce the same hash and share a directory.

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

### 8.3 Manifest List Exactness
- When N snapshots behind AND write overlap exists, read exactly N historical manifest lists (I/O convoy)
- Not N-1, not N+1
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

### 8.12 Cross-Table Retry Cost
- CAS failures caused by commits to a different table (or disjoint partitions of the same table) do not require manifest I/O
- The retry cost is: 1 catalog read + 1 CAS round-trip
- Per-attempt I/O and conflict resolution costs apply ONLY when `has_write_overlap()` returns True
- This is critical for multi-table workloads: with N tables and uniform selection, ~(N-1)/N of CAS failures are cross-table and essentially free to retry

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
- **Per-attempt cost**: I/O paid on first attempt and on retries with write overlap (ML read, MF write, ML write)
- **Conflict cost**: Additional I/O paid only on retry with write overlap (type-dependent)
