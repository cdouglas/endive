# Plan: Integrating Accurate Storage Distributions

This plan describes the remaining work to fully integrate the June 2025 YCSB benchmark
data into the simulator and ensure the simulator correctly models real-world storage.

## Current Status

### Completed
1. **Provider profiles corrected** - Separated storage tiers:
   - `s3`: AWS S3 Standard (CAS only)
   - `s3x`: AWS S3 Express One Zone (CAS + append)
   - `azure`: Azure Blob Storage (CAS + append)
   - `azurex`: Azure Premium Block Blob (CAS + append)
   - `gcp`: GCP Cloud Storage (CAS only)
   - `instant`: Hypothetical fast system
   - `aws`: Backward compatibility alias for `s3x`

2. **Basic tests updated** - Tests validate profiles match measurements

### Remaining Work

## Phase 1: Analysis Pipeline Updates

### 1.1 Parameter Extraction
Add new parameters to `extract_key_parameters()` in `saturation_analysis.py`:

```python
# In extract_key_parameters():
if 'storage' in config:
    params['storage_provider'] = config['storage'].get('provider', None)

if 'catalog' in config:
    params['table_metadata_inlined'] = config['catalog'].get('table_metadata_inlined', True)
    params['catalog_backend'] = config['catalog'].get('backend', 'storage')

if 'transaction' in config:
    params['manifest_list_mode'] = config['transaction'].get('manifest_list_mode', 'rewrite')
```

### 1.2 Grouping Support
Ensure analysis can group by:
- `storage_provider` (s3, s3x, azure, azurex, gcp)
- `table_metadata_inlined` (true, false)
- `manifest_list_mode` (rewrite, append)

## Phase 2: Per-Thread Distribution Support

The YCSB data includes per-thread distribution fits. Currently we use aggregate fits.

### 2.1 Add Per-Thread Profiles
Extend PROVIDER_PROFILES to include per-thread parameters:

```python
"s3x": {
    "cas": {
        "median": 22, "sigma": 0.22,  # Aggregate (current)
        "per_thread": {
            1: {"mu": 9.585, "sigma": 0.057},
            4: {"mu": 9.893, "sigma": 0.187},
            8: {"mu": 10.014, "sigma": 0.150},
            16: {"mu": 10.128, "sigma": 0.213},
        }
    },
    ...
}
```

### 2.2 Concurrency-Aware Latency
Modify `get_cas_latency()` to interpolate between per-thread parameters based on
current contention level (from CONTENTION_TRACKER).

## Phase 3: Failure Latency Distributions

Currently we use a single `failure_multiplier`. The data shows failures have
distinct distributions, not just scaled success latencies.

### 3.1 Separate Failure Distributions
Add failure-specific distribution parameters:

```python
"azure": {
    "append": {
        "success": {"median": 87, "sigma": 0.28},
        "failure": {"median": 3011, "sigma": 0.5},  # 31.6x slower, different shape
    }
}
```

### 3.2 Update Latency Functions
Modify `get_cas_latency(success)` and `get_append_latency(success)` to use
separate distributions rather than multipliers.

## Phase 4: Comprehensive Validation Tests

### 4.1 Distribution Conformance Tests
For each provider, test that generated samples match expected percentiles:

```python
class TestDistributionConformance:
    """Validate simulated distributions match YCSB measurements."""

    @pytest.mark.parametrize("provider,op,expected_p50,expected_p99", [
        ("s3", "cas", 60.8, 103),
        ("s3x", "cas", 22.4, 44.3),
        ("s3x", "append", 20.5, 48.2),
        ("azure", "cas", 93.1, 4700),  # Very heavy tail
        ("azure", "append", 87.3, 207),
        ("azurex", "cas", 63.5, 4099),
        ("azurex", "append", 69.9, 101),
        ("gcp", "cas", 170, 6546),
    ])
    def test_percentiles_match_measurements(self, provider, op, expected_p50, expected_p99):
        # Generate samples with provider profile
        # Assert percentiles within tolerance
        pass
```

### 4.2 K-S Test Validation
Use Kolmogorov-Smirnov tests to validate distribution shape:

```python
def test_ks_statistic_acceptable(self, provider, op):
    """Generated samples should pass K-S test against expected lognormal."""
    # Generate samples
    # Perform K-S test against lognormal(mu, sigma)
    # Assert KS statistic < threshold (e.g., 0.1)
```

### 4.3 Contention Scaling Tests
Validate latency increases appropriately with concurrent operations:

```python
def test_contention_scaling_matches_measurements(self, provider):
    """Latency at 16 threads should match measured scaling factor."""
    # Simulate with 1 concurrent op -> measure latency
    # Simulate with 16 concurrent ops -> measure latency
    # Assert ratio matches provider's contention_scaling factor
```

### 4.4 Failure Latency Tests
Validate failure latencies match measurements:

```python
def test_azure_append_failure_latency(self):
    """Azure append failures should be ~31x slower than successes."""
    # Generate success latencies
    # Generate failure latencies
    # Assert ratio matches failure_multiplier
```

## Phase 5: Experiment Configurations

### 5.1 Provider Comparison Experiments
Create configs for comparing providers:

```
exp7_1_s3_vs_s3x.toml        # S3 Standard vs Express
exp7_2_azure_vs_azurex.toml  # Azure Blob vs Premium
exp7_3_cross_cloud.toml      # S3x vs Azure vs GCP
```

### 5.2 Feature Experiments
Create configs for testing table metadata inlining and ML+ mode:

```
exp8_1_metadata_inlining_s3x.toml    # Impact of inlining on S3 Express
exp8_2_metadata_inlining_azure.toml  # Impact on Azure (heavier tails)
exp8_3_ml_plus_s3x.toml              # ML+ mode with S3 Express
exp8_4_ml_plus_azure.toml            # ML+ mode with Azure
exp8_5_combined_s3x.toml             # Both inlining + ML+ on S3 Express
exp8_6_combined_azurex.toml          # Both on Azure Premium
```

### 5.3 Analysis Scripts
Update `regenerate_all_plots.sh` to include new experiment patterns.

## Phase 6: Documentation

### 6.1 Update CLAUDE.md
Document new provider names and their characteristics.

### 6.2 Update Errata
Record any simplifications or known limitations in `docs/append_errata.md`.

### 6.3 Update README
Add section on provider selection and storage tier considerations.

## Implementation Order

1. **Phase 1** (Analysis pipeline) - Required to run experiments
2. **Phase 4.1-4.2** (Basic validation tests) - Should run before experiments
3. **Phase 5.1-5.2** (Experiment configs) - Run experiments
4. **Phase 2** (Per-thread distributions) - Enhancement, not blocking
5. **Phase 3** (Failure distributions) - Enhancement, not blocking
6. **Phase 4.3-4.4** (Advanced validation) - After enhancements
7. **Phase 6** (Documentation) - Ongoing

## Test Matrix

| Provider | CAS | Append | Failure Mult | Contention | Per-Thread |
|----------|-----|--------|--------------|------------|------------|
| s3       | ✓   | ✗      | ✓            | ✓          | Pending    |
| s3x      | ✓   | ✓      | ✓            | ✓          | Pending    |
| azure    | ✓   | ✓      | ✓            | ✓          | Pending    |
| azurex   | ✓   | ✓      | ✓            | ✓          | Pending    |
| gcp      | ✓   | ✗      | ✓            | ✓          | Pending    |
| instant  | ✓   | ✓      | N/A          | N/A        | N/A        |

## Success Criteria

1. All provider profiles validated against YCSB measurements
2. K-S test statistic < 0.15 for all distributions
3. Percentiles within 30% of measured values
4. Contention scaling factors within 20% of measured
5. Failure multipliers correctly applied
6. Analysis pipeline can filter/group by new parameters
7. Experiment configs produce valid results
