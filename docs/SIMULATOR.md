# Simulator Internals

This document describes the Endive simulator in fine detail, including pseudocode and code snippets to verify the model against the codebase.

## Overview

Endive simulates Apache Iceberg's optimistic concurrency control (OCC) using SimPy discrete-event simulation. The key abstraction is **message-passing**: transactions never access catalog state directly; all state is obtained via explicit API calls (`catalog.read()`, `catalog.try_cas()`) that pay network latency.

## Core Data Structures

### CatalogSnapshot (endive/snapshot.py)

Immutable snapshot of catalog state at a specific time:

```python
@dataclass(frozen=True)
class CatalogSnapshot:
    seq: int                                      # Global sequence number
    tbl: tuple[int, ...]                          # Per-table version numbers
    partition_seq: tuple[tuple[int, ...], ...] | None  # Per-partition versions (2D)
    ml_offset: tuple[int, ...]                    # Manifest list byte offsets per table
    partition_ml_offset: tuple[tuple[int, ...], ...] | None  # Per-partition ML offsets
    timestamp: int                                # Simulation time when captured
```

### CASResult (endive/snapshot.py)

Result of a CAS operation with server-time snapshot:

```python
@dataclass(frozen=True)
class CASResult:
    success: bool                 # True if CAS succeeded
    snapshot: CatalogSnapshot     # State at server processing time
```

### Txn (endive/transaction.py)

Transaction state from creation through commit/abort:

```python
@dataclass
class Txn:
    id: int
    t_submit: int                 # Simulation time when submitted (ms)
    t_runtime: int                # Execution duration (ms)
    v_catalog_seq: int            # Catalog seq when snapshot taken
    v_tblr: dict[int, int]        # Table versions read {table_id: version}
    v_tblw: dict[int, int]        # Table versions to write {table_id: version}
    n_retries: int = 0            # Retry count after CAS failures
    t_commit: int = -1            # Commit time (-1 if not committed)
    t_abort: int = -1             # Abort time (-1 if not aborted)
    v_dirty: dict[int, int]       # Versions being validated

    # Partition mode fields
    partitions_read: dict[int, set[int]]           # table_id -> partition_ids read
    partitions_written: dict[int, set[int]]        # table_id -> partition_ids written
    v_partition_seq: dict[int, dict[int, int]]     # table_id -> partition_id -> version

    # Snapshot fields (message-passing)
    start_snapshot: CatalogSnapshot | None         # Initial snapshot from read()
    current_snapshot: CatalogSnapshot | None       # Latest snapshot (updated on retry)
```

### Catalog (endive/main.py)

Server-side catalog state:

```python
class Catalog:
    def __init__(self, sim):
        self.sim = sim
        self.seq = 0                              # Global sequence number
        self.tbl = [0] * N_TABLES                 # Per-table versions
        self.ml_offset = [0] * N_TABLES           # Per-table manifest list sizes

        # Partition mode (when PARTITION_ENABLED)
        self.partition_seq = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
        self.partition_ml_offset = [[0] * N_PARTITIONS for _ in range(N_TABLES)]
```

## Transaction Lifecycle

### 1. Transaction Generation (txn_gen)

```python
def txn_gen(sim, txn_id, catalog):
    # 1. Read catalog state (PAYS ROUND-TRIP LATENCY)
    start_snapshot = yield from catalog.read()

    # 2. Select tables using snapshot (not live catalog)
    tblr, tblw = rand_tbl_from_snapshot(start_snapshot)

    # 3. Sample runtime from lognormal distribution
    t_runtime = T_MIN_RUNTIME + np.random.lognormal(T_RUNTIME_MU, T_RUNTIME_SIGMA)

    # 4. Create transaction with snapshot-derived state
    txn = Txn(txn_id, sim.now, t_runtime, start_snapshot.seq, tblr, tblw)
    txn.v_dirty = {**tblr, **tblw}
    txn.start_snapshot = start_snapshot
    txn.current_snapshot = start_snapshot

    # 5. Partition mode: select partitions and capture per-partition versions
    if PARTITION_ENABLED:
        for tbl_id in tblr.keys():
            parts_r, parts_w = select_partitions(N_PARTITIONS)
            txn.partitions_read[tbl_id] = parts_r
            if tbl_id in tblw:
                txn.partitions_written[tbl_id] = parts_w
            # Capture partition versions FROM SNAPSHOT
            for p in parts_r:
                txn.v_partition_seq[tbl_id][p] = start_snapshot.partition_seq[tbl_id][p]

    # 6. Execute transaction (simulated work)
    yield sim.timeout(txn.t_runtime)

    # 7. Commit loop
    while txn.t_commit < 0 and txn.t_abort < 0:
        txn.n_retries += 1

        # Optional backoff (if configured)
        if txn.n_retries > 1 and BACKOFF_ENABLED:
            backoff = calculate_backoff_time(txn.n_retries - 1)
            yield sim.timeout(backoff)

        yield sim.process(txn_commit(sim, txn, catalog))
```

**Key insight**: The snapshot is captured BEFORE `yield sim.timeout(txn.t_runtime)`. This means `n_behind` (snapshots behind) grows proportionally to runtime.

### 2. Catalog Read (message-passing)

```python
def read(self):
    """Read catalog state - pays full round-trip latency."""
    yield self.sim.timeout(get_cas_latency())  # Network RTT
    return self._create_snapshot()             # Returns immutable snapshot
```

### 3. Catalog CAS (message-passing)

```python
def try_cas(self, txn, expected_snapshot):
    """CAS with server-time snapshot capture."""
    # Request to server (half RTT)
    yield self.sim.timeout(get_cas_latency() / 2)

    # Server captures snapshot BEFORE mutation
    server_snapshot = self._create_snapshot()

    # Perform CAS check
    success = self.try_CAS(self.sim, txn)

    # Response back (half RTT)
    yield self.sim.timeout(get_cas_latency() / 2)

    return CASResult(success=success, snapshot=server_snapshot)
```

## CAS Conflict Detection

### Partition Mode (PARTITION_ENABLED=True)

```python
def _try_CAS_partition(self, sim, txn):
    """Vector clock comparison at partition granularity."""
    # Check all touched partitions
    for tbl_id, partitions in txn.partitions_read.items():
        for p in partitions:
            if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                return False  # Conflict!

    for tbl_id, partitions in txn.partitions_written.items():
        for p in partitions:
            if self.partition_seq[tbl_id][p] != txn.v_partition_seq[tbl_id][p]:
                return False  # Conflict!

    # Success: increment written partition versions
    for tbl_id, partitions in txn.partitions_written.items():
        for p in partitions:
            self.partition_seq[tbl_id][p] += 1
    self.seq += 1
    return True
```

### Table-Level Mode (N_GROUPS == N_TABLES)

```python
def try_CAS(self, sim, txn):
    for t in txn.v_dirty.keys():
        if self.tbl[t] != txn.v_dirty[t]:
            return False  # Conflict!

    # Success: update table versions
    for t, v in txn.v_tblw.items():
        self.tbl[t] = v
    self.seq += 1
    return True
```

### Catalog-Level Mode (default)

```python
def try_CAS(self, sim, txn):
    if self.seq != txn.v_catalog_seq:
        return False  # Any concurrent commit is a conflict

    for t, v in txn.v_tblw.items():
        self.tbl[t] = v
    self.seq += 1
    return True
```

## Conflict Resolution

### txn_commit Flow

```python
def txn_commit(sim, txn, catalog):
    # 1. Attempt CAS via message-passing API
    cas_result = yield from catalog.try_cas(txn, txn.current_snapshot)
    server_snapshot = cas_result.snapshot

    if cas_result.success:
        txn.t_commit = sim.now
        return  # Done!

    # 2. CAS failed - check retry limit
    if txn.n_retries >= N_TXN_RETRY:
        txn.t_abort = sim.now
        return

    # 3. Resolve conflicts based on mode
    if PARTITION_ENABLED:
        # Partition mode: resolve per-partition
        yield from resolver.merge_partition_conflicts(sim, txn, catalog, partition_seq_snapshot)
    else:
        # Table mode: read historical manifest lists
        n_snapshots_behind = server_snapshot.seq - txn.v_catalog_seq
        yield from resolver.read_manifest_lists(sim, n_snapshots_behind, txn.id)
        yield from resolver.merge_table_conflicts(sim, txn, v_catalog, catalog, server_snapshot)

    # 4. Read fresh snapshot for retry
    retry_snapshot = yield from catalog.read()
    txn.current_snapshot = retry_snapshot
    txn.v_catalog_seq = retry_snapshot.seq
```

### Historical Manifest List Reading (Table Mode)

**CRITICAL**: On CAS failure, must read manifest lists from ALL intermediate snapshots:

```python
def read_manifest_lists(sim, n_snapshots: int, txn_id: int, avg_ml_size: int = None):
    """Read n manifest lists in batches of MAX_PARALLEL."""
    if n_snapshots <= 0:
        return

    for batch_start in range(0, n_snapshots, MAX_PARALLEL):
        batch_size = min(MAX_PARALLEL, n_snapshots - batch_start)

        # Parallel reads - take max latency
        if T_PUT is not None and avg_ml_size is not None:
            batch_latencies = [get_put_latency(avg_ml_size) for _ in range(batch_size)]
        else:
            batch_latencies = [get_manifest_list_latency('read') for _ in range(batch_size)]

        yield sim.timeout(max(batch_latencies))
```

**Example**: Transaction 150s behind at 25 commits/sec per partition:
- `n_behind = 150 * 25 = 3,750` manifest lists to read
- Batches = ceil(3750 / 4) = 938
- Time = 938 * 30ms = **28 seconds** (using PUT latency)

### Partition Conflict Resolution

```python
def merge_partition_conflicts(sim, txn, catalog, partition_seq_snapshot):
    """Resolve conflicts for each conflicting partition."""
    conflicting = get_conflicting_partitions(txn, partition_seq_snapshot)

    for tbl_id, partitions in conflicting.items():
        for p in partitions:
            # Calculate snapshots behind FOR THIS PARTITION
            snapshot_v = partition_seq_snapshot[tbl_id][p]
            txn_v = txn.v_partition_seq[tbl_id][p]
            n_behind = snapshot_v - txn_v

            # Read partition's historical manifest lists
            avg_ml_size = catalog.partition_ml_offset[tbl_id][p]
            yield from read_partition_manifest_lists(sim, n_behind, txn.id, tbl_id, p, avg_ml_size)

            # Resolve based on conflict type
            is_real = np.random.random() < REAL_CONFLICT_PROBABILITY
            if is_real:
                yield from resolve_partition_real_conflict(sim, txn, tbl_id, p, catalog)
            else:
                yield from resolve_partition_false_conflict(sim, txn, tbl_id, p, catalog)
```

### False Conflict Resolution

```python
def resolve_false_conflict(sim, txn, table_id, v_catalog, catalog, snapshot):
    """No data overlap - just need to merge manifest list pointers."""

    # Read metadata root
    yield sim.timeout(get_metadata_root_latency('read'))

    if not TABLE_METADATA_INLINED:
        yield sim.timeout(get_table_metadata_latency('read'))

    if MANIFEST_LIST_MODE == "append":
        # ML+ mode: tentative entry still valid, no ML update needed
        pass
    else:
        # Rewrite mode: must read and rewrite manifest list
        ml_size = snapshot.ml_offset[table_id] if snapshot else None

        # Read ML
        if ml_size and T_PUT:
            yield sim.timeout(get_put_latency(ml_size))
        else:
            yield sim.timeout(get_manifest_list_latency('read'))

        # Write new ML
        if ml_size and T_PUT:
            yield sim.timeout(get_manifest_list_write_latency(ml_size))
        else:
            yield sim.timeout(get_manifest_list_latency('write'))
```

### Real Conflict Resolution

```python
def resolve_real_conflict(sim, txn, table_id, v_catalog, catalog, snapshot):
    """Data overlap - must read/write manifest FILES."""

    # Sample number of conflicting manifest files
    n_manifests = sample_conflicting_manifests()

    # Read metadata root and manifest list
    yield sim.timeout(get_metadata_root_latency('read'))
    yield sim.timeout(get_manifest_list_latency('read'))

    # Read manifest files in parallel batches
    for batch_start in range(0, n_manifests, MAX_PARALLEL):
        batch_size = min(MAX_PARALLEL, n_manifests - batch_start)
        batch_latencies = [get_manifest_file_latency('read') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    # Write merged manifest files
    for batch_start in range(0, n_manifests, MAX_PARALLEL):
        batch_size = min(MAX_PARALLEL, n_manifests - batch_start)
        batch_latencies = [get_manifest_file_latency('write') for _ in range(batch_size)]
        yield sim.timeout(max(batch_latencies))

    # Write updated manifest list
    yield sim.timeout(get_manifest_list_latency('write'))
```

## Latency Models

### CAS Latency

```python
def get_cas_latency(success: bool = True) -> float:
    """Lognormal distribution with separate success/failure parameters."""
    if not success and 'failure' in T_CAS:
        config = T_CAS['failure']
    else:
        config = T_CAS

    latency = np.random.lognormal(mean=config['mu'], sigma=config['sigma'])
    return max(latency, MIN_LATENCY)
```

### PUT Latency (size-based)

```python
def get_put_latency(size_bytes: int) -> float:
    """Durner et al. model: latency = base + size_mib * rate + noise."""
    base_latency = T_PUT['base_latency_ms']      # e.g., 30ms
    latency_per_mib = T_PUT['latency_per_mib_ms']  # e.g., 20ms/MiB
    sigma = T_PUT['sigma']                        # e.g., 0.3

    size_mib = size_bytes / (1024 * 1024)
    deterministic = base_latency + size_mib * latency_per_mib

    mu = np.log(deterministic)
    latency = np.random.lognormal(mean=mu, sigma=sigma)
    return max(latency, MIN_LATENCY)
```

### Manifest List Latency

```python
def get_manifest_list_latency(operation: str) -> float:
    """Fixed lognormal distribution for ML read/write."""
    params = T_MANIFEST_LIST[operation]  # 'read' or 'write'
    return np.random.lognormal(mean=params['mu'], sigma=params['sigma'])
```

## Storage Provider Latencies

| Provider | CAS Median | ML Read | ML Write | PUT Base | PUT Rate |
|----------|------------|---------|----------|----------|----------|
| S3 | 61ms | 61ms | 63ms | 30ms | 20 ms/MiB |
| S3 Express | 22ms | 22ms | 21ms | 10ms | 10 ms/MiB |
| Azure | 93ms | 93ms | 95ms | 50ms | 25 ms/MiB |
| Azure Premium | 64ms | 64ms | 70ms | 30ms | 15 ms/MiB |
| Instant | 1ms | 1ms | 1ms | 0.5ms | 0.1 ms/MiB |

## Critical Invariants

1. **Snapshot captured BEFORE runtime**: `v_partition_seq` is set when transaction is created, not when it commits. This means `n_behind` grows with runtime.

2. **Historical ML reading scales with n_behind**: Each retry reads `n_behind` manifest lists. For long-running transactions, this dominates commit latency.

3. **PUT latency used when T_PUT configured**: When `T_PUT` is set and `avg_ml_size` is provided, ML reads use `get_put_latency()` (~30ms) instead of `get_manifest_list_latency()` (~60ms).

4. **Retry updates snapshot**: After each failed CAS, the transaction reads a fresh snapshot. The second retry's `n_behind` is much smaller (only commits during the first retry).

5. **Partition mode is per-partition, not per-table**: Each partition has its own version counter. A transaction touching partition 0 doesn't conflict with one touching partition 1.

## Commit Latency Breakdown Example

For a transaction with 150s runtime at 25 commits/sec per partition:

```
Initial n_behind = 150 * 25 = 3,750 snapshots

Retry 1:
  - Read 3,750 MLs in batches of 4: ceil(3750/4) * 30ms = 28.1s
  - False conflict resolution: ~60ms
  - CAS attempt: ~1ms
  Total: ~28.2s

During retry 1, new commits: 28.2s * 25/s = 705

Retry 2:
  - Read 705 MLs: ceil(705/4) * 30ms = 5.3s
  - False conflict resolution: ~60ms
  - CAS attempt (success): ~1ms
  Total: ~5.4s

Total commit latency: 28.2 + 5.4 = 33.6s
```

This matches observed P50 of ~34s for 2-partition experiments at high throughput.
