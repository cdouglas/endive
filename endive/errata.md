# Errata: Bugs Found During Refactoring

This document records bugs found during the refactoring of `main.py` into modules.
These bugs are **deferred** until after refactoring is complete to maintain exact
determinism verification.

## Baseline

- Test suite: 387 tests passing
- Determinism verified: 34 transactions, mean retries 1.029, mean latency 99.53ms
- Baseline config: `/tmp/baseline_test.toml`

## Progress

### Completed Phases
- Phase 1: Baseline established
- Phase 2: Extract utils.py (git functions, partition_tables_into_groups)
- Phase 3: Extract config.py (PROVIDER_PROFILES, latency parsing, validation)

### Current Stats
- main.py: 3263 → 2532 lines (-731 lines)
- config.py: 598 lines
- utils.py: 124 lines
- Total: 3254 lines (vs 3263 original)

### Deferred Phases
Phases 4-9 are deferred due to global state dependencies:
- Latency functions use 15+ global variables
- Classes (Catalog, ConflictResolver) depend on globals and latency functions
- Python's import binding prevents simple extraction

### Options for Remaining Refactoring
1. **Config namespace**: Convert globals to `config.VAR` pattern (requires updating all references)
2. **Dependency injection**: Pass config as parameter to classes (requires changing constructors)
3. **Keep as-is**: Accept that some code stays in main.py

## Bugs Found

_None found - code is functionally identical_

## Format

When adding bugs, use this format:

### BUG-001: Brief Description
- **Found in**: Phase N - module.py
- **Location**: `function_name()` around line X
- **Description**: What the bug is
- **Impact**: How it affects simulation
- **Fix deferred because**: Fixing would change determinism / require test changes
