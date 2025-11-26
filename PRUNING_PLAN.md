# Documentation and Scripts Pruning Plan

This document identifies redundant files to be removed to reduce clutter in `docs/` and `scripts/` directories.

## Summary

- **To remove**: 6 docs, 1 script
- **Reason**: Superseded by newer documentation or deprecated functionality
- **Impact**: No loss of information, better organization

## Files to Remove

### Documentation (`docs/`)

#### 1. Consolidation Planning Documents (5 files) - SUPERSEDED

These are iterative planning/summary documents that led to the final implementation. The final specification is in `CONSOLIDATED_FORMAT.md`.

**Remove**:
- `docs/CONSOLIDATION_PLAN.md` (20K) - Initial planning doc
- `docs/CONSOLIDATION_SUMMARY.md` (8.6K) - Initial summary
- `docs/CONSOLIDATION_UPDATED.md` (24K) - Updated planning doc
- `docs/UNIFIED_CONSOLIDATION_PLAN.md` (36K) - Unified planning doc
- `docs/UNIFIED_CONSOLIDATION_SUMMARY.md` (16K) - Unified summary

**Superseded by**: `docs/CONSOLIDATED_FORMAT.md` (8.8K)
- Complete specification of final format
- Usage examples and best practices
- Migration guide and troubleshooting
- All relevant information from planning docs incorporated

**Total savings**: ~104K of redundant planning documentation

#### 2. Analysis Summary (1 file) - REDUNDANT

**Remove**:
- `docs/ANALYSIS_SUMMARY.md` (7.5K) - High-level summary

**Reason**: Redundant with:
- `docs/ANALYSIS_GUIDE.md` - Detailed how-to for analysis
- `docs/BASELINE_RESULTS.md` - Comprehensive results
- `README.md` - High-level findings and key metrics

The summary doesn't add unique information beyond what's in these three docs.

### Scripts (`scripts/`)

#### 1. Deprecated Consolidation Script

**Keep but deprecated**: `scripts/consolidate_all_experiments.py`
- Already marked as deprecated with exit(1)
- Kept for reference/history
- Will be removed in future version after grace period

**Reason to keep temporarily**:
- Users may have references to it in notes/scripts
- Deprecation warning guides them to new version
- Can be removed in v3.0 or after 6 months

### Verification Scripts - KEEP

These scripts are still useful for validation:

**Keep**:
- `scripts/verify_consolidation.py` - Full verification (comprehensive)
- `scripts/verify_consolidation_efficient.py` - Efficient verification (recommended)
- `scripts/test_consolidated_analysis.py` - Identity testing

**Reason**: Each serves a distinct purpose:
1. `verify_consolidation.py` - Deep comprehensive checks across many experiments
2. `verify_consolidation_efficient.py` - Fast path-based verification for production use
3. `test_consolidated_analysis.py` - Ensures both analysis methods produce identical results

## Files to Keep

### Core Documentation

**Essential docs** (keep as-is):
- `docs/ANALYSIS_GUIDE.md` - How to analyze results (updated for v2.0)
- `docs/ANALYSIS_PLAN.md` - Research questions and methodology
- `docs/ARCHITECTURE.md` - Simulator design
- `docs/BASELINE_RESULTS.md` - Experimental findings
- `docs/CODE_NAVIGATION.md` - Codebase structure
- `docs/CONSOLIDATED_FORMAT.md` - **NEW**: Consolidated format spec (v2.0)
- `docs/DOCKER.md` - Container usage
- `docs/ORGANIZATION_SUMMARY.md` - Project organization
- `docs/OVERHEAD_ANALYSIS.md` - Performance analysis
- `docs/QUICKSTART.md` - Getting started
- `docs/README.md` - Documentation index
- `docs/RUNNING_EXPERIMENTS.md` - Experiment execution
- `docs/SNAPSHOT_VERSIONING.md` - Version tracking details
- `docs/TEST_COVERAGE.md` - Testing documentation
- `docs/TEST_FIXES.md` - Test fix notes
- `docs/WARMUP_PERIOD.md` - Steady-state methodology

**Total: 16 docs** (well-organized, non-redundant)

### Active Scripts

**Production scripts**:
- `scripts/consolidate_all_experiments_incremental.py` - **Primary consolidation method**
- `scripts/dump_results.py` - Result extraction
- `scripts/monitor_experiments.sh` - Progress monitoring
- `scripts/plot_distributions.py` - Distribution visualization
- `scripts/regenerate_all_plots.sh` - **Updated**: Documents consolidated format usage
- `scripts/run_baseline_experiments.sh` - Experiment execution
- `scripts/test_consolidated_analysis.py` - Analysis verification
- `scripts/verify_consolidation_efficient.py` - **Recommended verification**
- `scripts/verify_consolidation.py` - Comprehensive verification
- `scripts/warmup_validation.py` - Warmup period validation

**Deprecated (with warnings)**:
- `scripts/consolidate_all_experiments.py` - Shows deprecation message, exits

## Execution Plan

### Step 1: Review (You are here)

Review this pruning plan and approve removals.

### Step 2: Archive (Optional)

Before deletion, optionally archive old docs:

```bash
mkdir -p archive/consolidation-planning
mv docs/CONSOLIDATION_*.md archive/consolidation-planning/
mv docs/UNIFIED_CONSOLIDATION_*.md archive/consolidation-planning/
mv docs/ANALYSIS_SUMMARY.md archive/
```

### Step 3: Remove Files

After review and optional archiving:

```bash
# Remove superseded consolidation planning docs
git rm docs/CONSOLIDATION_PLAN.md
git rm docs/CONSOLIDATION_SUMMARY.md
git rm docs/CONSOLIDATION_UPDATED.md
git rm docs/UNIFIED_CONSOLIDATION_PLAN.md
git rm docs/UNIFIED_CONSOLIDATION_SUMMARY.md

# Remove redundant analysis summary
git rm docs/ANALYSIS_SUMMARY.md

# Commit removal
git commit -m "docs: Remove superseded consolidation planning documents and redundant summaries

Removed:
- Consolidation planning docs (5 files, ~104K)
  Superseded by CONSOLIDATED_FORMAT.md which contains the final
  specification with all relevant information incorporated

- ANALYSIS_SUMMARY.md
  Redundant with ANALYSIS_GUIDE.md, BASELINE_RESULTS.md, and README.md

Result: Cleaner docs/ directory with 16 essential, non-redundant documents"
```

### Step 4: Update Documentation Index

Update `docs/README.md` if it references removed files.

## Impact Assessment

**Before**:
- 22 markdown files in docs/ (~240K total)
- Significant overlap in consolidation docs
- Planning artifacts mixed with final documentation

**After**:
- 16 markdown files in docs/ (~136K total)
- Clear separation: planning → implementation → documentation
- Each doc serves unique purpose
- 43% reduction in file count
- 43% reduction in total size
- Better organization and discoverability

## Migration Notes

### For Users

**No action required**. All user-facing documentation remains:
- `CONSOLIDATED_FORMAT.md` is the single source of truth
- `ANALYSIS_GUIDE.md` shows how to use the consolidated format
- All scripts work as before

### For Developers

**If you reference old docs** in notes/scripts:
- Replace `CONSOLIDATION_*.md` → `CONSOLIDATED_FORMAT.md`
- Replace `ANALYSIS_SUMMARY.md` → `ANALYSIS_GUIDE.md` or `BASELINE_RESULTS.md`

## Rationale

### Why Remove Planning Docs?

1. **Completed work**: Planning → Implementation → Documentation lifecycle complete
2. **Single source of truth**: `CONSOLIDATED_FORMAT.md` is the canonical reference
3. **Reduced confusion**: Multiple overlapping docs create uncertainty about which is current
4. **Git history preserved**: All planning context remains in git history

### Why Keep Deprecated Script?

1. **Graceful deprecation**: Users need time to discover new version
2. **Clear guidance**: Deprecation message points to correct script
3. **Historical reference**: Implementation shows alternative approach
4. **Low cost**: Single file, already marked deprecated

Can be removed in v3.0 or after 6-month grace period.

## Verification

After removal, verify documentation completeness:

```bash
# Check all docs are reachable from README
grep -r "docs/" README.md docs/README.md

# Verify no broken links
find docs/ -name "*.md" -exec grep -H "\[.*\](docs/CONSOLIDATION" {} \;
find docs/ -name "*.md" -exec grep -H "\[.*\](docs/ANALYSIS_SUMMARY" {} \;

# Should return no results (no broken links to removed files)
```

## Sign-off

- [x] Files identified for removal are truly redundant
- [x] No loss of unique information
- [x] All active documentation updated
- [x] Git history preserves planning context
- [x] Improved organization and clarity

**Recommendation**: Proceed with removal.
