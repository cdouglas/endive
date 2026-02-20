# Provider Separation Design

## Problem

The current simulator conflates catalog and storage latencies into a single "provider" configuration. This is architecturally incorrect:

- **Catalog**: Where the table pointer lives. Handles CAS operations for atomic updates.
- **Storage**: Where manifest lists, manifest files, and data files live.

These can be different systems with different latency characteristics:

| Scenario | Catalog | Storage | Use Case |
|----------|---------|---------|----------|
| Fast catalog | Nessie (1ms) | S3 Standard | Cost-optimized with fast commits |
| Unified fast | S3 Express | S3 Express | Performance-optimized |
| Hybrid | S3 Express | S3 Standard | Fast commits, cheap bulk storage |

## Current Issue

With `provider = "instant"`, both catalog CAS and storage I/O are 1ms, leading to:
- Unrealistically high throughput (50+ c/s)
- Conflict resolution is nearly free (~3ms)
- Results don't reflect production behavior

## Solution: Abstract Providers

### 1. CatalogProvider Interface

```python
class CatalogProvider(ABC):
    @abstractmethod
    def get_cas_latency(self) -> float:
        """Generate CAS latency in milliseconds."""
        pass

    @abstractmethod
    def supports_append(self) -> bool:
        """Whether catalog supports append operations."""
        pass
```

Implementations:
- `InstantCatalog`: 1ms CAS (models Nessie, Polaris)
- `ObjectStorageCatalog`: Uses object storage conditional writes (S3, Azure, GCS)

### 2. StorageProvider Interface

```python
class StorageProvider(ABC):
    @abstractmethod
    def get_manifest_list_read_latency(self) -> float:
        pass

    @abstractmethod
    def get_manifest_list_write_latency(self) -> float:
        pass

    # ... manifest file, PUT operations
```

### 3. Config Format

```toml
[catalog]
type = "instant"           # or "object_storage"
provider = "s3x"           # for object_storage type

[storage]
provider = "s3x"           # manifest list/file storage
```

## Integration Plan

### Phase 1: Add Provider Abstractions (done)
- `endive/catalog_provider.py`: CatalogProvider interface
- `endive/storage_provider.py`: StorageProvider interface

### Phase 2: Update main.py
Replace global latency functions with provider methods:

```python
# Before
def get_cas_latency():
    return generate_latency_lognormal(T_CAS_MEDIAN, T_CAS_SIGMA)

# After
CATALOG_PROVIDER: CatalogProvider = None

def get_cas_latency():
    return CATALOG_PROVIDER.get_cas_latency()
```

### Phase 3: Update Config Loading

```python
def configure_from_toml(path):
    global CATALOG_PROVIDER, STORAGE_PROVIDER

    catalog_cfg = config.get("catalog", {})
    catalog_type = catalog_cfg.get("type", "object_storage")

    if catalog_type == "instant":
        CATALOG_PROVIDER = InstantCatalog()
    else:
        provider = catalog_cfg.get("provider", storage_cfg.get("provider"))
        CATALOG_PROVIDER = ObjectStorageCatalog(provider)

    storage_provider = config["storage"]["provider"]
    STORAGE_PROVIDER = ObjectStorageProvider(storage_provider)
```

### Phase 4: Backward Compatibility

For configs without explicit `[catalog].type`:
- Use `[storage].provider` for both catalog and storage
- This matches current behavior

## Expected Results

With separated providers:

| Config | Catalog | Storage | Expected Throughput |
|--------|---------|---------|--------------------:|
| instant/instant | 1ms | 1ms | ~50 c/s (limited by arrival) |
| instant/s3x | 1ms | 22ms | ~45 c/s (CAS not bottleneck) |
| s3x/s3x | 22ms | 22ms | ~13 c/s (realistic) |
| s3/s3 | 61ms | 61ms | ~5 c/s (realistic) |

The key insight: **with instant catalog + realistic storage**, conflict resolution still costs ~87ms (ML read + ML write), but CAS is fast. This isolates the catalog contention effect from storage effects.
