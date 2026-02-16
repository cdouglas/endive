# Errata: Bugs Found During Refactoring

This document records bugs found during the refactoring of `main.py` into modules.
These bugs are **deferred** until after refactoring is complete to maintain exact
determinism verification.

## Baseline

- Test suite: 387 tests passing
- Determinism hash (seed=42, 60s simulation):
  ```
  329c3b05e0829ff45a9de4e12e76df7016e294517ee901451d48963aa28450a2
  ```
- Baseline config: `/tmp/baseline_test.toml`

## Bugs Found

_None yet - refactoring in progress_

## Format

When adding bugs, use this format:

### BUG-001: Brief Description
- **Found in**: Phase N - module.py
- **Location**: `function_name()` around line X
- **Description**: What the bug is
- **Impact**: How it affects simulation
- **Fix deferred because**: Fixing would change determinism / require test changes
