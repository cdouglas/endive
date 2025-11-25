# Test Fixes - min_seeds Filtering

## Issue

After implementing `min_seeds` filtering (default = 3) to ensure statistical robustness, 5 existing tests began failing because they created experiments with fewer than 3 seeds.

## Root Cause

The tests were written before `min_seeds` filtering was implemented. They created:
- 1 seed per experiment (3 tests)
- 2 seeds per experiment (1 test)
- 1 seed per experiment via CLI (3 tests)

With `min_seeds=3` (the default), these experiments were filtered out, resulting in empty dataframes and test failures.

## Failures

```
FAILED test_build_index_structure - assert 0 == 1
FAILED test_full_workflow - assert 0 == 2
FAILED test_cli_overrides_config_file - CLI failed (empty results)
FAILED test_cli_without_config_uses_defaults - CLI failed (empty results)
FAILED test_cli_group_by_override - CLI failed (empty results)
```

## Solution

### Strategy 1: Override min_seeds in CONFIG (2 tests)

For tests that use `build_experiment_index()` directly:

```python
# Before
index_df = build_experiment_index(tmpdir, "exp_test-*")

# After
original_config = saturation_analysis.CONFIG.copy()
try:
    saturation_analysis.CONFIG['analysis'] = saturation_analysis.CONFIG.get('analysis', {}).copy()
    saturation_analysis.CONFIG['analysis']['min_seeds'] = 1  # or 2
    index_df = build_experiment_index(tmpdir, "exp_test-*")
finally:
    saturation_analysis.CONFIG = original_config
```

**Applied to:**
- `test_build_index_structure` (set min_seeds=1)
- `test_full_workflow` (set min_seeds=2)

### Strategy 2: Add min_seeds to config file (2 tests)

For tests that use CLI with a config file:

```python
config_content = """
[analysis]
min_seeds = 1
"""
```

**Applied to:**
- `test_cli_overrides_config_file`
- `test_cli_group_by_override`

### Strategy 3: Create 3 seeds (1 test)

For tests that use CLI without a config file (relying on defaults):

```python
# Before: Create 1 seed
seed_dir = exp_dir / "12345"
seed_dir.mkdir()
# ... create data ...

# After: Create 3 seeds
for seed in ["12345", "23456", "34567"]:
    seed_dir = exp_dir / seed
    seed_dir.mkdir()
    # ... create data ...
```

**Applied to:**
- `test_cli_without_config_uses_defaults`

## Test Results

Before fixes: **5 failed, 31 passed**

After fixes: **36 passed** (test_saturation_analysis.py)

Combined with regression tests: **47 passed** (all tests)

## Design Decision: Keep min_seeds=3 Default

The default of `min_seeds=3` is intentional and correct:

1. **Statistical validity**: 3 seeds is the minimum for meaningful standard deviation
2. **Production use**: Real experiments should have ≥3 seeds for robustness
3. **Test clarity**: Tests that need <3 seeds should explicitly override this

The failing tests were edge cases (single-seed smoke tests) that should explicitly opt out of the min_seeds requirement.

## Lessons Learned

1. **Configuration defaults matter**: When adding new filtering/validation, review all tests
2. **Test isolation**: Tests should not rely on implicit defaults when testing edge cases
3. **Explicit is better**: Tests with unusual configurations (1-2 seeds) should make this explicit
4. **Guard against regressions**: The new regression test suite would have caught this

## Related Files Modified

- `tests/test_saturation_analysis.py` - Fixed 5 failing tests
- `tests/test_saturation_analysis_regression.py` - New tests properly set min_seeds where needed

## Verification

All tests pass:
```bash
pytest tests/test_saturation_analysis.py -v      # 36 passed
pytest tests/test_saturation_analysis_regression.py -v  # 11 passed
pytest tests/test_saturation_analysis*.py -v     # 47 passed total
```
