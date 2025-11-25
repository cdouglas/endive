# Test Coverage for Saturation Analysis

## Overview

This document summarizes the test coverage for the saturation analysis module, with emphasis on regression tests that detect measurement accuracy issues.

## Test Files

### `tests/test_saturation_analysis.py`
Original test suite covering:
- Experiment directory scanning
- Parameter extraction from configs
- Results loading and aggregation across seeds
- Basic statistics computation
- Configuration system (TOML files and CLI overrides)

**Coverage**: 36 tests, focusing on functional correctness and configuration handling

### `tests/test_saturation_analysis_regression.py` (NEW)
Regression test suite specifically designed to catch measurement accuracy issues.

**Coverage**: 11 tests across 5 test classes

## Regression Tests (New)

### 1. TestThroughputCalculationAccuracy (4 tests)

**Purpose**: Detect errors in throughput calculation, especially after warmup/cooldown filtering

#### `test_throughput_with_warmup_cooldown_filtering`
- **Catches**: Duration calculated as `max(t_submit)` instead of `(max - min)`
- **Method**: Creates 1-hour simulation with known transaction rate
- **Validates**: Measured throughput matches expected rate in active window
- **Would detect**: The original regression (5× error in throughput)

#### `test_duration_calculation_with_offset_timestamps`
- **Catches**: Duration calculation bug directly
- **Method**: Uses dataframe with timestamps offset from 0 (like after filtering)
- **Validates**: Duration is `(max - min)`, not `max`
- **Critical**: This is the most direct test of the bug fix

#### `test_per_seed_throughput_with_filtering`
- **Catches**: Per-seed statistics also using wrong duration calculation
- **Method**: Creates multiple seeds with known rates
- **Validates**: Each seed's throughput is calculated correctly

#### `test_throughput_independent_of_k_min_cycles`
- **Catches**: Measurement window size affecting throughput (should be invariant)
- **Method**: Tests with different K_MIN_CYCLES values (3, 5, 7)
- **Validates**: Throughput is an intensive property - independent of measurement window size
- **Fundamental sanity check**: If this fails, something is fundamentally wrong

### 2. TestWarmupCooldownFiltering (2 tests)

**Purpose**: Verify warmup/cooldown filtering works correctly

#### `test_warmup_computation_uses_config`
- **Catches**: CONFIG values not being used in warmup calculation
- **Validates**: Warmup duration respects k_min_cycles, min/max bounds

#### `test_transactions_outside_window_excluded`
- **Catches**: Transactions in warmup/cooldown periods being included in statistics
- **Validates**: Only transactions in active window are counted

### 3. TestStatisticalAccuracy (3 tests)

**Purpose**: Verify computed statistics accurately reflect the data

#### `test_overhead_percentage_calculation`
- **Validates**: Overhead = (commit_latency / total_latency) × 100

#### `test_success_rate_with_mixed_status`
- **Validates**: Success rate correctly handles committed vs aborted transactions

#### `test_retry_statistics_only_for_committed`
- **Catches**: Mean retries calculated over all transactions instead of committed only
- **Important**: Clarifies semantic meaning of statistics

### 4. TestEndToEndAccuracy (1 test)

**Purpose**: Integration test verifying complete analysis pipeline

#### `test_full_experiment_analysis_accuracy`
- **Validates**: All statistics (throughput, success rate, latencies, overhead) are accurate
- **Method**: Creates realistic experiment with known properties
- **Comprehensive**: Tests entire pipeline from data loading to index building

### 5. TestRegressionDetection (1 test)

**Purpose**: Meta-test verifying these tests would catch known regressions

#### `test_would_catch_max_instead_of_max_minus_min_bug`
- **Method**: Simulates the original bug manually
- **Validates**: Test assertions would fail with buggy calculation
- **Confidence check**: Proves these tests are effective

## Test Effectiveness

### Bugs These Tests Would Catch

1. **Throughput calculation errors** (primary regression)
   - Using `max` instead of `(max - min)` for duration
   - Off-by-factor errors in throughput
   - Duration calculated over wrong window

2. **Warmup/cooldown filtering errors**
   - Transactions outside active window included
   - CONFIG values not respected
   - Filtering applied incorrectly

3. **Statistical calculation errors**
   - Overhead percentage formula wrong
   - Success rate calculation wrong
   - Mean retries calculated over wrong population

4. **Measurement invariance violations**
   - Throughput varying with measurement window size
   - Statistics sensitive to filtering parameters when they shouldn't be

### Running the Tests

```bash
# Run all saturation analysis tests
pytest tests/test_saturation_analysis*.py -v

# Run only regression tests
pytest tests/test_saturation_analysis_regression.py -v

# Run specific test class
pytest tests/test_saturation_analysis_regression.py::TestThroughputCalculationAccuracy -v

# Run with output
pytest tests/test_saturation_analysis_regression.py -v -s
```

### Test Philosophy

These regression tests follow key principles:

1. **Measure what you claim to measure**: If we say we're measuring steady-state throughput, verify the measurement is accurate

2. **Test with realistic scenarios**: Use 1-hour simulations, realistic warmup/cooldown periods, actual filtering logic

3. **Known inputs → expected outputs**: Create data with known properties, verify computed statistics match

4. **Test edge cases**: Offset timestamps, filtered windows, multiple seeds

5. **Verify invariants**: Properties that should be constant (e.g., throughput independent of measurement window size)

6. **Meta-validation**: Test that tests would catch known bugs

## Coverage Gaps (Future Work)

Areas that could use additional tests:

1. **Plot generation**: Tests verify data loading/statistics, but not plot rendering
2. **Markdown table generation**: Format and content correctness
3. **Group-by functionality**: Experiments grouped by parameters
4. **Error handling**: Malformed data, missing files, corrupt parquets
5. **Performance**: Large experiments with many seeds
6. **Commit rate over time**: Time-series analysis accuracy

## Continuous Integration

To prevent regressions, these tests should run:
- On every commit
- Before merging pull requests
- After dependency updates
- As part of release validation

**Critical**: The regression tests in `test_saturation_analysis_regression.py` are specifically designed to catch the throughput calculation bug that caused the 5× discrepancy. They should never be disabled or relaxed without careful review.
