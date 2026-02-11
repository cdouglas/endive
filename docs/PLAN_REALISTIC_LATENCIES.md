# Plan: Realistic CAS and Append Latency Modeling

This plan improves simulation accuracy based on real-world measurements from AWS, Azure, and GCP (June 2025 YCSB benchmarks).

## Executive Summary

The current simulator uses **normal distributions** for storage latencies, but real-world measurements show latencies are **lognormally distributed** with significant provider-specific variation. This plan:

1. Switches to lognormal latency distributions
2. Separates storage configuration from catalog configuration
3. Makes explicit which components live in storage vs catalog services
4. Adds cloud provider profiles with measured parameters
5. Models separate latency distributions for success vs failure paths
6. Adds contention-based latency scaling
7. Creates validation tests ensuring simulator output matches expected bounds

---

## Key Findings from Measurements

### Latency Distribution Shape
All measurements fit lognormal distributions:

| Provider | Operation | mu | sigma | Median (ms) | Mean (ms) |
|----------|-----------|------|-------|-------------|-----------|
| AWS | CAS | 10.16 | 0.45 | 22.9 | 29.2 |
| AWS | Append | 9.90 | 0.25 | 20.5 | 20.7 |
| Azure | CAS | 11.44 | 0.80 | 74.7 | 268 |
| Azure | Append | 11.25 | 0.28 | 76.9 | 81.3 |
| GCP | CAS | 12.44 | 0.91 | 170 | 530 |

### Tail Latency (p99/median ratios)
- AWS CAS: 3.4x (moderate tails)
- AWS Append: 2.3x (light tails)
- Azure CAS: 57.5x (heavy tails!)
- Azure Append: 2.1x (light tails)
- GCP CAS: 38.6x (heavy tails)

### Failed vs Successful Operation Latency
Critical insight: failed operations have different latencies:

| Provider | Operation | Success Mean | Fail Mean | Ratio |
|----------|-----------|--------------|-----------|-------|
| AWS | CAS | 29.2 ms | 34.1 ms | 1.17x |
| AWS | Append | 20.7 ms | 22.6 ms | 1.09x |
| Azure | CAS | 268 ms | 274 ms | 1.02x |
| Azure | Append | 81.3 ms | 2787 ms | **34.3x** |
| GCP | CAS | 530 ms | 7111 ms | **13.4x** |

### Contention-Based Latency Scaling
Latency increases with concurrent writers:

| Provider | Operation | 1 thread | 16 threads | Scaling |
|----------|-----------|----------|------------|---------|
| AWS | CAS | 24.0 ms | 32.6 ms | 1.4x |
| AWS | Append | 13.1 ms | 24.2 ms | 1.8x |
| Azure | CAS | 77.3 ms | 446 ms | 5.8x |
| Azure | Append | 79.5 ms | 83.5 ms | 1.1x |

---

## Architecture: Storage vs Catalog

### Component Location Model

The simulator models several components that can reside in different systems:

| Component | Location Options | Notes |
|-----------|------------------|-------|
| **Manifest list** | Always storage | S3/GCS/ADLS - immutable files |
| **Manifest files** | Always storage | S3/GCS/ADLS - immutable files |
| **Catalog pointer (CAS)** | Storage OR catalog service | Hadoop catalog = storage; REST/Nessie = service |
| **Intention log (append)** | Storage OR FIFO queue | Depends on append protocol design |
| **Checkpoints** | Always storage | Compaction writes to storage |

### Why This Matters

1. **Catalog in storage** (Hadoop catalog): CAS/append operations use S3 conditional PUT/append. Latencies are storage latencies.

2. **Catalog as service** (REST catalog, Nessie): CAS operations go to a catalog service with its own latency profile. May be faster or slower than storage.

3. **FIFO queue for intentions**: Append operations go to a fast queue (Kafka, SQS). Only checkpoints touch storage.

The configuration must be explicit about where each operation goes, so the correct latency distributions are applied.

---

## Implementation Plan

### Phase 1: Lognormal Latency Distribution

**File: `endive/main.py`**

#### 1.1 Add lognormal latency generator

Replace `generate_latency()` (line ~554) which uses normal distribution:

```python
def generate_latency(mean: float, stddev: float) -> float:
    """Generate storage operation latency from normal distribution."""
    return max(MIN_LATENCY, np.random.normal(loc=mean, scale=stddev))
```

With lognormal-based generator:

```python
def generate_latency_lognormal(mu: float, sigma: float) -> float:
    """Generate storage operation latency from lognormal distribution.

    Args:
        mu: Mean of underlying normal distribution (log-scale)
        sigma: Std dev of underlying normal distribution (log-scale)

    Returns:
        Latency in milliseconds
    """
    return max(MIN_LATENCY, np.random.lognormal(mean=mu, sigma=sigma))


def lognormal_params_from_median_sigma(median_ms: float, sigma: float) -> tuple[float, float]:
    """Convert median + sigma to lognormal mu parameter.

    For lognormal, median = exp(mu), so mu = ln(median).

    Args:
        median_ms: Desired median latency in milliseconds
        sigma: Shape parameter (higher = heavier tail)

    Returns:
        (mu, sigma) for np.random.lognormal()
    """
    mu = np.log(median_ms)
    return mu, sigma
```

#### 1.2 Backward compatibility for latency configuration

Support both legacy (mean/stddev) and new (median/sigma) formats:
- If `distribution = "lognormal"`: use median/sigma directly
- If only `mean/stddev`: convert using `mu = ln(mean) - sigma²/2` approximation
- Legacy configs continue to work unchanged

---

### Phase 2: Separate Storage and Catalog Configuration

**File: `endive/main.py`**

#### 2.1 New configuration schema

The configuration now has two top-level sections for latencies:

```toml
[storage]
# Controls latencies for operations that ALWAYS go to storage:
# - Manifest list read/write
# - Manifest file read/write
# - Table metadata read/write (when inlined metadata disabled)
# - Checkpoints (compaction output)

provider = "aws"  # Sets baseline storage latencies from measurements

# Explicit overrides (optional)
T_MANIFEST_LIST.read.median = 50
T_MANIFEST_LIST.read.sigma = 0.3


[catalog]
# Controls catalog commit protocol
mode = "cas"  # or "append"

# WHERE does the catalog live?
backend = "storage"      # Catalog ops use [storage] latencies
# OR
backend = "service"      # Catalog ops use [catalog.service] latencies
# OR
backend = "fifo_queue"   # Appends go to queue, checkpoints to storage
```

#### 2.2 Catalog backend configurations

**Backend: `storage`** (Hadoop-style catalog)
```toml
[catalog]
mode = "cas"
backend = "storage"

[catalog.storage]
# Optional: override storage provider just for catalog
# If omitted, inherits from [storage].provider
provider = "aws"

# Optional: explicit overrides
T_CAS.median = 25.0
T_CAS.sigma = 0.5
```

**Backend: `service`** (REST catalog, Nessie)
```toml
[catalog]
mode = "cas"
backend = "service"

[catalog.service]
# Catalog service latency profile
provider = "aws"  # Use AWS-like latencies for catalog service

# OR explicit configuration:
T_CAS.median = 15.0
T_CAS.sigma = 0.3
T_CAS.failure_multiplier = 1.2
contention_scaling = 1.2
```

**Backend: `fifo_queue`** (Append to queue, checkpoint to storage)
```toml
[catalog]
mode = "append"
backend = "fifo_queue"

[catalog.fifo_queue]
# Queue append is typically very fast
append.median = 5.0
append.sigma = 0.2
append.failure_multiplier = 1.1

# Checkpoints still go to storage
checkpoint_backend = "storage"  # Uses [storage] latencies
```

#### 2.3 Provider profiles

Built-in profiles based on measurements:

```python
PROVIDER_PROFILES = {
    "aws": {
        # Storage operations (manifest list/file, CAS when catalog in storage)
        "manifest_list": {"read": {"median": 50, "sigma": 0.3},
                          "write": {"median": 60, "sigma": 0.3}},
        "manifest_file": {"read": {"median": 50, "sigma": 0.3},
                          "write": {"median": 60, "sigma": 0.3}},
        # Catalog operations (when backend = "storage")
        "cas": {"median": 23, "sigma": 0.45, "failure_multiplier": 1.17},
        "append": {"median": 20, "sigma": 0.25, "failure_multiplier": 1.09},
        "contention_scaling": {"cas": 1.4, "append": 1.8},
    },
    "azure": {
        "manifest_list": {"read": {"median": 75, "sigma": 0.3},
                          "write": {"median": 85, "sigma": 0.3}},
        "manifest_file": {"read": {"median": 75, "sigma": 0.3},
                          "write": {"median": 85, "sigma": 0.3}},
        "cas": {"median": 75, "sigma": 0.80, "failure_multiplier": 1.02},
        "append": {"median": 77, "sigma": 0.28, "failure_multiplier": 34.3},
        "contention_scaling": {"cas": 5.8, "append": 1.1},
    },
    "gcp": {
        "manifest_list": {"read": {"median": 100, "sigma": 0.5},
                          "write": {"median": 120, "sigma": 0.5}},
        "manifest_file": {"read": {"median": 100, "sigma": 0.5},
                          "write": {"median": 120, "sigma": 0.5}},
        "cas": {"median": 170, "sigma": 0.91, "failure_multiplier": 13.4},
        "append": None,  # No append data available
        "contention_scaling": {"cas": 0.7, "append": None},
    },
    "instant": {
        # Hypothetical infinitely fast system
        "manifest_list": {"read": {"median": 1, "sigma": 0.1},
                          "write": {"median": 1, "sigma": 0.1}},
        "manifest_file": {"read": {"median": 1, "sigma": 0.1},
                          "write": {"median": 1, "sigma": 0.1}},
        "cas": {"median": 1, "sigma": 0.1, "failure_multiplier": 1.0},
        "append": {"median": 1, "sigma": 0.1, "failure_multiplier": 1.0},
        "contention_scaling": {"cas": 1.0, "append": 1.0},
    },
}
```

---

### Phase 3: Configuration Precedence Rules

**Resolution order** (later overrides earlier):

1. Built-in hardcoded defaults
2. `[storage].provider` profile (for storage operations)
3. `[catalog.X].provider` profile (for catalog operations, where X is backend type)
4. Explicit `T_*` parameters at any level

**Example resolution:**

```toml
[storage]
provider = "aws"                    # Sets manifest latencies from AWS profile
T_MANIFEST_LIST.read.median = 100   # Override just manifest list read

[catalog]
backend = "service"

[catalog.service]
provider = "azure"                  # Catalog service uses Azure-like latencies
T_CAS.median = 50                   # But override CAS median
```

Result:
- `T_MANIFEST_LIST.read`: median=100 (explicit override)
- `T_MANIFEST_LIST.write`: median=60 (AWS profile default)
- `T_MANIFEST_FILE.*`: AWS profile defaults
- `T_CAS`: median=50 (explicit), sigma=0.80 (Azure profile)
- `T_APPEND`: Azure profile defaults

---

### Phase 4: Failure Latency and Contention Scaling

**File: `endive/main.py`**

#### 4.1 Modify latency functions

Update `get_cas_latency()` and `get_append_latency()` to accept success/failure context:

```python
def get_catalog_latency(op_type: str, success: bool = True) -> float:
    """Get catalog operation latency.

    Args:
        op_type: "cas" or "append"
        success: If True, draw from success distribution; else failure distribution

    Returns:
        Latency in milliseconds
    """
    config = CATALOG_LATENCY_CONFIG[op_type]
    base = generate_latency_lognormal(config['mu'], config['sigma'])

    # Apply contention scaling
    if CONTENTION_SCALING_ENABLED:
        factor = get_contention_factor(op_type)
        base *= factor

    # Apply failure multiplier
    if not success:
        base *= config.get('failure_multiplier', 1.0)

    return base
```

#### 4.2 Track concurrent operations

```python
class ContentionTracker:
    """Track concurrent catalog operations for latency scaling."""
    def __init__(self):
        self.active_cas = 0
        self.active_append = 0

    def get_contention_factor(self, op_type: str) -> float:
        """Get latency multiplier based on current contention level."""
        if op_type == "cas":
            n = max(1, self.active_cas)
            scaling = CATALOG_CONTENTION_SCALING.get('cas', 1.0)
        else:
            n = max(1, self.active_append)
            scaling = CATALOG_CONTENTION_SCALING.get('append', 1.0)

        # Linear interpolation: factor = 1 + (scaling - 1) * (n - 1) / 15
        return 1.0 + (scaling - 1.0) * min(n - 1, 15) / 15.0
```

---

### Phase 5: Validation Tests

**File: `tests/test_realistic_latencies.py`**

#### 5.1 Test lognormal distribution shape

```python
class TestLognormalLatencyDistribution:
    """Validate that simulated latencies match lognormal distribution."""

    def test_cas_latency_percentiles_aws_storage(self):
        """Test CAS latency with catalog in AWS storage."""
        # Config: storage.provider=aws, catalog.backend=storage
        # Verify p50≈23ms, p95≈65ms, p99≈78ms (within 20%)

    def test_append_latency_percentiles_aws_storage(self):
        """Test append latency with catalog in AWS storage."""
        # p50≈20ms, p95≈28ms, p99≈48ms

    def test_lognormal_ks_test(self):
        """Verify latency samples fit lognormal distribution (K-S test)."""
        # K-S statistic < 0.1
```

#### 5.2 Test storage vs catalog separation

```python
class TestStorageCatalogSeparation:
    """Validate storage and catalog configurations are independent."""

    def test_storage_aws_catalog_instant(self):
        """AWS storage latencies with instant catalog."""
        # storage.provider=aws, catalog.backend=service, catalog.service.provider=instant
        # Manifest ops should be slow (~50ms), CAS should be fast (~1ms)

    def test_storage_instant_catalog_azure(self):
        """Instant storage with Azure catalog latencies."""
        # storage.provider=instant, catalog.backend=service, catalog.service.provider=azure
        # Manifest ops should be fast (~1ms), CAS should be slow (~75ms)

    def test_fifo_queue_fast_append_slow_checkpoint(self):
        """FIFO queue appends fast, checkpoints use storage latency."""
        # catalog.backend=fifo_queue, catalog.fifo_queue.append.median=5
        # storage.provider=azure
        # Appends ~5ms, checkpoints ~75ms
```

#### 5.3 Test failure latency behavior

```python
class TestFailureLatencyModeling:
    """Validate failure latency characteristics."""

    def test_aws_failure_slightly_slower(self):
        """AWS failures should be ~1.2x slower than successes."""
        # Ratio should be 1.0-1.5x

    def test_azure_append_failure_much_slower(self):
        """Azure append failures should be >>10x slower than successes."""
        # This is the dramatic 34x ratio observed
        # Only applies when catalog.backend=storage with Azure provider
```

#### 5.4 Integration test: end-to-end simulation bounds

```python
class TestSimulationBounds:
    """End-to-end tests that simulation stays within realistic bounds."""

    def test_aws_storage_catalog_throughput(self):
        """With AWS storage+catalog, max CAS throughput ~30-35 ops/sec."""

    def test_instant_catalog_aws_storage_throughput(self):
        """With instant catalog + AWS storage, throughput limited by conflict resolution."""
        # Should be higher than pure AWS, but still bounded by manifest I/O

    def test_append_mode_vs_cas_mode(self):
        """Append mode should achieve higher throughput than CAS."""
        # AWS append ~90 ops/sec vs CAS ~32 ops/sec when catalog in storage
```

---

## Configuration Examples

### Example 1: Hadoop Catalog (everything in S3)
```toml
[storage]
provider = "aws"

[catalog]
mode = "cas"
backend = "storage"
# CAS uses S3 conditional PUT with AWS latency profile
# Result: T_CAS drawn from AWS storage measurements (median=23ms)
```

### Example 2: REST Catalog + S3 Storage
```toml
[storage]
provider = "aws"  # Manifests in S3

[catalog]
mode = "cas"
backend = "service"

[catalog.service]
# REST catalog is faster than S3 CAS
T_CAS.median = 15.0
T_CAS.sigma = 0.3
```

### Example 3: Append Mode with FIFO Queue
```toml
[storage]
provider = "aws"  # Manifests in S3

[catalog]
mode = "append"
backend = "fifo_queue"

[catalog.fifo_queue]
append.median = 5.0       # Queue is fast
append.sigma = 0.2
checkpoint_backend = "storage"  # Compaction writes to S3
```

### Example 4: "What if Catalog Were Infinitely Fast?"
```toml
[storage]
provider = "aws"  # Realistic storage

[catalog]
mode = "cas"
backend = "service"

[catalog.service]
provider = "instant"  # Near-zero catalog latency
# Tests: where is the bottleneck when catalog isn't it?
```

### Example 5: Azure Storage, AWS-hosted REST Catalog
```toml
[storage]
provider = "azure"  # Manifests in ADLS (slow, heavy tails)

[catalog]
mode = "cas"
backend = "service"

[catalog.service]
provider = "aws"  # REST catalog on AWS (faster, lighter tails)
```

### Example 6: Simulate "What if Azure Append Failures Were Fast?"
```toml
[storage]
provider = "azure"

[catalog]
mode = "append"
backend = "storage"

[catalog.storage]
# Use Azure defaults but override the extreme failure multiplier
T_APPEND.failure_multiplier = 1.1  # Instead of 34.3x
```

### Example 7: Legacy Configuration (Backward Compatible)
```toml
# Old-style config still works
[storage]
T_CAS.mean = 100
T_CAS.stddev = 10
T_MANIFEST_LIST.read.mean = 50
T_MANIFEST_LIST.read.stddev = 5

[catalog]
mode = "cas"
# backend defaults to "storage" for backward compatibility
# mean/stddev auto-converted to lognormal approximation
```

---

## What Each Provider Profile Contains

| Profile Section | Provides | Used When |
|-----------------|----------|-----------|
| `storage.provider` | `T_MANIFEST_LIST.*`, `T_MANIFEST_FILE.*` | Always (manifests always in storage) |
| `storage.provider` | `T_CAS.*`, `T_APPEND.*` | `catalog.backend = "storage"` |
| `catalog.service.provider` | `T_CAS.*`, `T_APPEND.*` | `catalog.backend = "service"` |
| `catalog.fifo_queue.*` | Queue append latency | `catalog.backend = "fifo_queue"` |
| `storage.provider` | Checkpoint latency | `catalog.backend = "fifo_queue"` (checkpoints) |

---

## Test Matrix

| Test | Configuration | Validates |
|------|---------------|-----------|
| `test_cas_percentiles_aws_storage` | storage=aws, catalog.backend=storage | p50/p95/p99 within 20% of AWS CAS |
| `test_append_percentiles_aws_storage` | storage=aws, catalog.backend=storage | p50/p95/p99 within 20% of AWS append |
| `test_manifest_latency_azure` | storage=azure | Manifest ops use Azure profile |
| `test_catalog_service_independent` | storage=aws, catalog.service=instant | CAS fast, manifests slow |
| `test_fifo_queue_separation` | catalog.backend=fifo_queue | Append fast, checkpoint slow |
| `test_failure_multiplier_aws` | storage=aws | fail/success 1.0-1.5x |
| `test_failure_multiplier_azure_append` | storage=azure, catalog.backend=storage | fail/success > 10x |
| `test_contention_scaling` | storage=aws, high load | Latency increases 1.2-2x |
| `test_throughput_ceiling_aws` | storage=aws, catalog.backend=storage | 25-40 CAS ops/sec |

---

## Implementation Order

1. **Phase 1** (Lognormal distribution): Core change, enables all other phases
2. **Phase 2** (Storage/catalog separation): New config schema with explicit backends
3. **Phase 3** (Precedence rules): Provider profiles as defaults, explicit overrides
4. **Phase 5** (Tests): Add tests early to validate phases 1-3
5. **Phase 4** (Failure latency & contention): Refinement for accuracy

---

## Files Modified

| File | Changes |
|------|---------|
| `endive/main.py` | Lognormal generator, storage/catalog config parsing, provider profiles, failure latency, contention tracking |
| `endive/capstats.py` | Export measured latency parameters for analysis |
| `tests/test_realistic_latencies.py` | New test file |
| `experiment_configs/` | New example configs demonstrating storage/catalog combinations |
| `docs/APPENDIX_SIMULATOR_DETAILS.md` | Document new latency model and configuration |

---

## Success Criteria

1. **Distribution Shape**: K-S test confirms lognormal fit (statistic < 0.1)
2. **Percentile Accuracy**: Simulated p50/p95/p99 within 20% of measured values
3. **Separation Correctness**: Storage and catalog latencies independently configurable
4. **Throughput Bounds**: Max throughput matches measured ceiling (±20%)
5. **Failure Behavior**: Failure latency ratio matches measurements (±50%)
6. **Backward Compatibility**: Existing configs produce valid results (auto-conversion to lognormal)

---

## Migration Path

1. **Existing configs continue to work**: `mean/stddev` format auto-converts to lognormal
2. **Default backend**: If `[catalog].backend` not specified, defaults to `"storage"` for backward compatibility
3. **Deprecation warnings**: Emit warning when using legacy `mean/stddev` format, recommend migration to `median/sigma`
4. **Documentation**: Update all experiment configs to use new format with explicit backends
