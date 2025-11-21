# Iceberg Catalog Simulator - Critical Context

## Architecture Overview

**Purpose:** Discrete-event simulation of Apache Iceberg-style optimistic concurrency control to explore commit latency tradeoffs in shared-storage catalog formats.

**Core Components:**
- **Catalog:** Versioned state with monotonically increasing sequence number (seq)
- **Transactions:** Capture snapshot version at start, attempt CAS at commit
- **Conflict Resolution:** Read manifest lists, merge conflicts, retry with updated version

## Key Invariants

### Version Tracking (CRITICAL)
1. Transaction captures `catalog.seq` as `v_catalog_seq` at creation (S_i)
2. On commit attempt, checks against current `catalog.seq` (S_{i+n})
3. If CAS fails with n = catalog.seq - v_catalog_seq:
   - Read EXACTLY n manifest lists (one per intermediate snapshot)
   - Update v_catalog_seq to current catalog.seq
   - Attempt to install v_catalog_seq + 1
4. Catalog.seq advances by exactly 1 on each successful commit
5. Table versions never decrease

### Table Grouping (CRITICAL)
1. Tables partitioned into G groups (1 ≤ G ≤ T)
2. Transactions NEVER span group boundaries
3. When G=1: catalog-level conflicts (any concurrent writes conflict)
4. When G=T: table-level conflicts (only same-table writes conflict)
5. Group assignment is deterministic (controlled by seed)

### Conflict Detection Modes
- **Catalog-level (G=1):** Compare catalog sequence numbers
- **Table-level (G=T):** Compare individual table versions in v_dirty
- **Group-level (1<G<T):** Catalog-level within group boundaries

## Storage Model (Iceberg on Object Stores)

### Latency Characteristics
- Normal distributions with mean ± stddev
- MIN_LATENCY enforced to prevent unrealistic zeros
- Separate read/write latencies for each operation type:
  - T_CAS: Catalog pointer swap (atomic operation)
  - T_METADATA_ROOT: Metadata file read/write
  - T_MANIFEST_LIST: Manifest list file read/write
  - T_MANIFEST_FILE: Manifest file read/write

### Parallelism Model
- MAX_PARALLEL limits concurrent manifest list reads during conflict resolution
- Batch processing: reads MAX_PARALLEL lists at a time, takes max(latencies)
- Models S3's parallel read capabilities with practical limits

### Realistic Parameter Ranges
- **Fast storage (S3 Express, local SSD):** mean=10-50ms, stddev=5-10ms
- **Standard cloud storage (S3):** mean=50-150ms, stddev=20-40ms
- **Slower storage (cross-region):** mean=200-500ms, stddev=50-100ms
- **min_latency:** 1-2ms (very fast) to 5-10ms (typical)

## Simulation Fidelity

### Accurately Modeled
1. Optimistic concurrency control with CAS
2. Snapshot isolation semantics
3. Manifest list reading on conflict (n lists for n snapshots behind)
4. Stochastic storage latencies
5. Parallel I/O with limits
6. Retry logic with backoff

### Simplified
1. Actual data merging (simulated with latency)
2. Manifest file structure (not explicitly modeled)
3. Network failures/retries (deterministic latencies)
4. Garbage collection/compaction
5. Read queries (only write transactions)

### Parameterized
1. Storage latencies (configurable distributions)
2. Workload patterns (Zipf distributions for table access)
3. Transaction runtime (lognormal distribution)
4. Client arrival patterns (exponential, uniform, normal, fixed)

## Critical Code Locations

### Version Tracking
- Snapshot capture: `icecap/main.py:527` (txn_gen creates Txn with catalog.seq)
- CAS check: `icecap/main.py:359-393` (Catalog.try_CAS)
- Manifest list reading: `icecap/main.py:443-452` (batch processing)
- Version update: `icecap/main.py:437` (txn.v_catalog_seq = catalog.seq)

### Table Grouping
- Partitioning: `icecap/main.py:52-131` (partition_tables_into_groups)
- Table selection: `icecap/main.py:460-497` (rand_tbl respects group boundaries)
- Conflict detection: `icecap/main.py:369-384` (table-level when G=T)

### Configuration
- Loading: `icecap/main.py:134-240` (configure_from_toml)
- Validation: Currently missing - should be added
- Defaults: num_groups=1 (catalog-level), min_latency=5ms

## Testing Strategy

### Test Organization (29 tests total)
1. **Core simulator (13 tests):** Determinism, parameter effects
2. **Conflict resolution (7 tests):** Latencies, parallelism, stochastic behavior
3. **Table grouping (9 tests):** Partitioning, transaction boundaries, determinism
4. **Snapshot versioning (7 tests):** Version capture, CAS, manifest reading, retries

### Key Test Patterns
- Determinism: Same seed → identical results
- Parameter effects: Verify expected relationships (more load → more retries)
- Boundary conditions: G=1 (catalog-level), G=T (table-level)
- Multi-retry scenarios: Verify version progression through multiple failures

## Common Pitfalls to Avoid

1. **Don't break determinism:** All randomness must use np.random with seed control
2. **Group boundaries are sacred:** Never let transactions span groups
3. **Version monotonicity:** catalog.seq and tbl[i] versions never decrease
4. **Manifest list count:** Must read EXACTLY n lists for n snapshots behind
5. **MIN_LATENCY:** Always enforce minimum to prevent unrealistic zeros
6. **Parallel I/O:** Batch operations respect MAX_PARALLEL limit

## Future Enhancement Opportunities

1. **Configuration validation** before simulation runs
2. **Event logging** for detailed post-simulation analysis
3. **Storage model abstraction** for different backend characteristics
4. **Workload/system separation** for clearer architecture
5. **Builder pattern** for test fixtures
6. **MODEL_FIDELITY.md** documenting what's accurate vs simplified

This context is essential for maintaining simulation accuracy when modifying the code or adding new features.
