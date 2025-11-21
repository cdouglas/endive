# Snapshot Versioning Verification

This document verifies that the simulator correctly implements snapshot-based optimistic concurrency control with proper version tracking through retries.

## Specification

The simulator must implement the following snapshot versioning logic:

1. **Transaction Start (S_i)**: Transaction captures snapshot version `S_i` when it reads the catalog
2. **Commit Attempt**: Transaction attempts CAS against current catalog version `S_{i+n}` for some `n ≥ 0`
3. **On CAS Failure (n > 0)**:
   - Read exactly `n` manifest lists (one per intermediate snapshot)
   - Merge conflicts
   - Attempt to install `S_{i+n+1}`
4. **On Repeated Failure**: If catalog moved to `S_{i+n+k}` for some `k > 0`, repeat with `k` manifest list reads

## Code Verification

### 1. Transaction Captures Initial Snapshot

**Location**: `icecap/main.py:527`

```python
def txn_gen(sim, txn_id, catalog):
    tblr, tblw = rand_tbl(catalog)
    t_runtime = T_MIN_RUNTIME + np.random.lognormal(...)
    # Transaction captures catalog.seq at creation time
    txn = Txn(txn_id, sim.now, int(t_runtime), catalog.seq, tblr, tblw)
    ...
```

**Verified**: Transaction captures `catalog.seq` as `txn.v_catalog_seq` at creation time (S_i).

### 2. CAS Checks Version Match

**Location**: `icecap/main.py:359-393` (Catalog.try_CAS)

For catalog-level conflicts (N_GROUPS = 1):
```python
def try_CAS(self, sim, txn):
    ...
    if self.seq == txn.v_catalog_seq:  # Check if versions match
        for off, val in txn.v_tblw.items():
            self.tbl[off] = val
        self.seq += 1  # Advance to next snapshot
        return True
    return False
```

For table-level conflicts (N_GROUPS = N_TABLES):
```python
    if N_GROUPS == N_TABLES:
        conflict = False
        for t in txn.v_dirty.keys():
            if self.tbl[t] != txn.v_dirty[t]:  # Check table versions
                conflict = True
                break
        if not conflict:
            # Commit and advance
            ...
            return True
        return False
```

**Verified**: CAS compares transaction's captured version against current catalog state.

### 3. Manifest List Reading on CAS Failure

**Location**: `icecap/main.py:424-452`

```python
def txn_commit(sim, txn, catalog):
    yield sim.timeout(get_cas_latency())

    if catalog.try_CAS(sim, txn):
        # Success
        ...
    else:
        # Calculate how many snapshots behind
        n_snapshots_behind = catalog.seq - txn.v_catalog_seq

        # Read n manifest lists for all snapshots between read and current
        if n_snapshots_behind > 0:
            # Process in batches of MAX_PARALLEL
            for batch_start in range(0, n_snapshots_behind, MAX_PARALLEL):
                batch_size = min(MAX_PARALLEL, n_snapshots_behind - batch_start)
                batch_latencies = [get_manifest_list_latency('read') for _ in range(batch_size)]
                yield sim.timeout(max(batch_latencies))
```

**Verified**:
- Correctly calculates `n = catalog.seq - txn.v_catalog_seq`
- Reads exactly `n` manifest lists
- Processes with parallelism limit MAX_PARALLEL

### 4. Version Update for Next Retry

**Location**: `icecap/main.py:437-480`

```python
    # Update to current catalog state
    v_catalog = dict()
    txn.v_catalog_seq = catalog.seq  # Update to S_{i+n}
    for t in txn.v_dirty.keys():
        v_catalog[t] = catalog.tbl[t]

    # ... conflict resolution ...

    # Update write set to the next available version per table
    for t in txn.v_tblw.keys():
        txn.v_tblw[t] = v_catalog[t] + 1  # Attempt to install S_{i+n+1}
```

**Verified**:
- Updates `txn.v_catalog_seq` to current `catalog.seq` (S_{i+n})
- Prepares to install S_{i+n+1} by setting write versions to current + 1

### 5. Retry Loop Handles Repeated Failures

**Location**: `icecap/main.py:534-537`

```python
def txn_gen(sim, txn_id, catalog):
    ...
    while txn.t_commit < 0 and txn.t_abort < 0:
        txn.n_retries += 1
        yield sim.process(txn_commit(sim, txn, catalog))
```

**Verified**: Retry loop continues until commit succeeds or transaction aborts, with version tracking maintained across retries.

## Test Validation

### Test Suite: `tests/test_snapshot_versioning.py`

All 7 tests pass, validating:

#### 1. TestSnapshotCapture
- ✅ `test_transaction_captures_initial_catalog_seq`
  - Verifies transaction captures `catalog.seq` at creation
  - Verifies captured version persists even as catalog advances

#### 2. TestCASVersionChecking
- ✅ `test_cas_succeeds_when_versions_match`
  - Verifies CAS succeeds when `txn.v_catalog_seq == catalog.seq`
  - Verifies catalog advances on success

- ✅ `test_cas_fails_when_versions_differ`
  - Verifies CAS fails when versions differ
  - Verifies catalog doesn't advance on failure

#### 3. TestManifestListReading
- ✅ `test_reads_n_manifest_lists_when_n_snapshots_behind`
  - Transaction at S_0, catalog at S_5
  - Verifies exactly 5 manifest lists are read
  - Validates `n_snapshots_behind = 5`

#### 4. TestRetryVersionProgression
- ✅ `test_retry_updates_to_current_version`
  - Transaction starts at S_10, catalog advances to S_15
  - After retry, verifies `txn.v_catalog_seq == 15`

- ✅ `test_multiple_retries_track_versions_correctly`
  - **Retry 1**: S_0 → S_3 (3 snapshots behind, reads 3 manifest lists)
  - **Retry 2**: S_3 → S_7 (4 snapshots behind, reads 4 manifest lists)
  - **Retry 3**: S_7 → S_9 (2 snapshots behind, reads 2 manifest lists)
  - Validates version progression: [0, 3, 7, 9]
  - Validates snapshots behind: [3, 4, 2]

#### 5. TestEndToEndVersioning
- ✅ `test_full_transaction_version_lifecycle`
  - Full simulation with high contention
  - Validates transactions with multiple retries
  - Confirms max retries observed: 9

## Example Trace

### Scenario: Transaction with Multiple Retries

**Initial State:**
```
Catalog: S_10
Transaction T1 starts, captures v_catalog_seq = 10
```

**First Commit Attempt:**
```
Catalog advanced to: S_13 (3 commits occurred)
T1 attempts CAS with v_catalog_seq = 10
CAS fails: 10 ≠ 13

n_snapshots_behind = 13 - 10 = 3
→ Read 3 manifest lists (for S_11, S_12, S_13)
→ Merge conflicts
→ Update txn.v_catalog_seq = 13
→ Prepare to install S_14
```

**Second Commit Attempt:**
```
Catalog advanced to: S_15 (2 more commits occurred)
T1 attempts CAS with v_catalog_seq = 13
CAS fails: 13 ≠ 15

n_snapshots_behind = 15 - 13 = 2
→ Read 2 manifest lists (for S_14, S_15)
→ Merge conflicts
→ Update txn.v_catalog_seq = 15
→ Prepare to install S_16
```

**Third Commit Attempt:**
```
Catalog still at: S_15 (no new commits)
T1 attempts CAS with v_catalog_seq = 15
CAS succeeds: 15 == 15
→ Install S_16
→ Catalog advances to S_16
→ Transaction commits successfully
```

## Verification Summary

### ✅ All Requirements Met

1. ✅ Transaction captures S_i at start
2. ✅ CAS checks against S_{i+n}
3. ✅ Reads exactly n manifest lists on failure
4. ✅ Attempts to install S_{i+n+1}
5. ✅ Repeats correctly for S_{i+n+k} with k manifest list reads
6. ✅ All behaviors validated by comprehensive tests

### Test Results

```
tests/test_snapshot_versioning.py::TestSnapshotCapture::
  test_transaction_captures_initial_catalog_seq PASSED

tests/test_snapshot_versioning.py::TestCASVersionChecking::
  test_cas_succeeds_when_versions_match PASSED
  test_cas_fails_when_versions_differ PASSED

tests/test_snapshot_versioning.py::TestManifestListReading::
  test_reads_n_manifest_lists_when_n_snapshots_behind PASSED

tests/test_snapshot_versioning.py::TestRetryVersionProgression::
  test_retry_updates_to_current_version PASSED
  test_multiple_retries_track_versions_correctly PASSED

tests/test_snapshot_versioning.py::TestEndToEndVersioning::
  test_full_transaction_version_lifecycle PASSED

7 passed in 0.65s
```

### Total Test Coverage

- **29 total tests pass** (13 original + 9 table grouping + 7 snapshot versioning)
- All snapshot versioning logic validated
- All table grouping logic validated
- All conflict resolution logic validated
- All determinism guarantees maintained

## Conclusion

The simulator correctly implements snapshot-based optimistic concurrency control with proper version tracking:

1. Transactions capture initial snapshot version at start
2. CAS operations validate version match before commit
3. Failed CAS triggers reading of exactly n manifest lists for n snapshots behind
4. Retries correctly update to current version and attempt next snapshot
5. Multiple retries handle arbitrary version progressions correctly

The implementation has been thoroughly validated through both code inspection and comprehensive automated tests.
