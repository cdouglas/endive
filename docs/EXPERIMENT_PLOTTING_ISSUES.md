# Experiment Plotting Issues - Action Required

This document outlines the 2 experiments that need fixes for correct plotting.

## Issue Summary

After reviewing all 10 experiments, **8 have correct plotting** but **2 need fixes**:

1. **exp3_2**: Mixes 4 distribution types on one plot
2. **exp5_3**: Mixes 6 CAS latencies on one plot

---

## Issue 1: exp3_2 (Manifest Count Distribution)

### Problem
The experiment sweeps 2 parameters but plots mix all configurations together:
- **Distribution type**: 4 variants (fixed-1, fixed-5, fixed-10, exponential)
- **Load** (inter_arrival.scale): 9 values

**Current plotting** (`regenerate_all_plots.sh` line 92):
```bash
"exp3_2_*|plots/exp3_2||Manifest distribution variance"
```
- No grouping or filtering
- All 36 experiments (4 × 9) plotted together
- **Cannot distinguish which distribution produces which results**

### Root Cause
The plot mixes experiments with fundamentally different conflict resolution costs:
- `fixed value=1`: 1 manifest file per conflict (best case)
- `fixed value=5`: 5 manifest files per conflict (typical)
- `fixed value=10`: 10 manifest files per conflict (worst case)
- `exponential mean=3`: Variable manifests with mean=3 (realistic)

These configurations are NOT comparable on the same axes.

### Solution

**Step 1**: Verify how distribution is encoded in data

First, check if there's a column that identifies the distribution type:
```bash
find experiments -name "experiment_index.csv" -path "*/exp3_2*" | head -1 | xargs head -2
```

Look for columns like:
- `conflicting_manifests_distribution`
- `conflicting_manifests_value`
- `conflicting_manifests_mean`

**Step 2**: Apply appropriate fix

**Option A - If distribution column exists**: Filter by distribution type
```bash
# Add to regenerate_all_plots.sh (replace line 92)

# Handle exp3.2 with filtering by distribution type
# Assuming these are the 4 distribution configs actually used:
declare -a exp3_2_dists=("fixed_1" "fixed_5" "fixed_10" "exponential")
declare -a exp3_2_labels=("fixed-1" "fixed-5" "fixed-10" "exponential")

for i in "${!exp3_2_dists[@]}"; do
    dist="${exp3_2_dists[$i]}"
    label="${exp3_2_labels[$i]}"

    # Wait for parallel slot...

    # Determine appropriate filter expression based on column names
    # Example if there's a "dist_type" column:
    run_analysis_with_filter \
        "exp3_2_*" \
        "plots/exp3_2_${dist}" \
        "" \
        "[FILTER_EXPRESSION_HERE]" \
        "Manifest distribution (${label})" &
    pids+=($!)
done
```

**Option B - If no distribution column**: Group by identifying parameter

If the distribution is identified by `conflicting_manifests_value` or `conflicting_manifests_mean`:
```bash
run_analysis \
    "exp3_2_*" \
    "plots/exp3_2" \
    "conflicting_manifests_value" \  # or whatever column exists
    "Manifest distribution variance"
```

**Step 3**: Update summary section (lines 372-376)
```bash
# Add exp3.2 plot locations
for dist_label in "fixed-1" "fixed-5" "fixed-10" "exponential"; do
    echo "  Manifest distribution (${dist_label}): plots/exp3_2_${dist_label}/"
done
```

---

## Issue 2: exp5_3 (Transaction Partitioning)

### Problem
The experiment sweeps 3 parameters but only groups by one:
- **CAS latency** (T_CAS): 6 values [15, 50, 100, 200, 500, 1000] ms
- **Partition count** (num_groups): 5 values [1, 2, 5, 10, 20]
- **Load** (inter_arrival.scale): 9 values

**Current plotting** (`regenerate_all_plots.sh` line 98):
```bash
"exp5_3_*|plots/exp5_3|num_groups|Transaction partitioning catalog latency"
```
- Groups by `num_groups` only
- All 270 experiments (6 × 5 × 9) mixed across 6 different CAS latencies
- **Cannot tell which data points come from which CAS latency**

### Root Cause
Comparing performance at 15ms CAS latency with 1000ms CAS latency on the same plot is meaningless. The CAS latency dramatically changes the performance characteristics.

### Solution

**Apply exp5_2 pattern** - Filter by CAS latency, group by num_groups:

```bash
# Replace line 98 in regenerate_all_plots.sh with:

# Handle exp5.3 with filtering by T_CAS value
# This generates 6 separate plots, one for each T_CAS value
declare -a exp5_3_t_cas_values=(15 50 100 200 500 1000)
for t_cas in "${exp5_3_t_cas_values[@]}"; do
    # Wait for parallel slot
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                unset 'pids[i]'
            fi
        done
        pids=("${pids[@]}")
        sleep 1
    done

    # Start analysis in background
    run_analysis_with_filter \
        "exp5_3_*" \
        "plots/exp5_3_t_cas_${t_cas}ms" \
        "num_groups" \
        "t_cas_mean==$t_cas" \
        "Transaction partitioning (T_CAS=${t_cas}ms)" &
    pids+=($!)
done
```

This generates:
- 6 plots (one per CAS latency)
- Each plot shows 5 series (one per num_groups value)
- Total: 6 plots × 5 series = all 30 (num_groups, CAS) combinations properly visualized

**Update summary section** (around line 394):

Replace:
```bash
echo "  Transaction partitioning catalog latency: plots/exp5_3/"
```

With:
```bash
# Add exp5.3 plot locations
for t_cas in "${exp5_3_t_cas_values[@]}"; do
    echo "  Transaction partitioning (T_CAS=${t_cas}ms): plots/exp5_3_t_cas_${t_cas}ms/"
done
```

**Update file counts section** (around line 406):

Add after exp5_2 file counts:
```bash
# Add exp5.3 file counts
for t_cas in "${exp5_3_t_cas_values[@]}"; do
    output_dir="plots/exp5_3_t_cas_${t_cas}ms"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
```

---

## Testing Fixes

After applying fixes:

### For exp3_2:
```bash
# Test with one distribution type
./scripts/regenerate_all_plots.sh --parallel 1
# Check that plots exist for each distribution
ls -la plots/exp3_2_*/
```

### For exp5_3:
```bash
# Regenerate plots
./scripts/regenerate_all_plots.sh --parallel 4

# Verify 6 plot directories were created
ls -d plots/exp5_3_t_cas_*ms/

# Check that each has the expected files
for dir in plots/exp5_3_t_cas_*ms/; do
    echo "$dir: $(find "$dir" -name "*.png" | wc -l) PNGs"
done
```

Expected output for exp5_3:
```
plots/exp5_3_t_cas_15ms/: 7 PNGs
plots/exp5_3_t_cas_50ms/: 7 PNGs
plots/exp5_3_t_cas_100ms/: 7 PNGs
plots/exp5_3_t_cas_200ms/: 7 PNGs
plots/exp5_3_t_cas_500ms/: 7 PNGs
plots/exp5_3_t_cas_1000ms/: 7 PNGs
```

---

## Priority

**exp5_3 is higher priority** because:
1. All 270 experiments have been run
2. The fix is straightforward (copy exp5_2 pattern)
3. Clear pattern already established

**exp3_2 requires investigation** first:
1. Need to verify distribution is encoded in data
2. May need additional columns exported during consolidation
3. Only 36 experiments, so less critical

---

## Checklist

- [x] **exp5_3**: Apply filtering by CAS latency
  - [x] Update regenerate_all_plots.sh (replace line 98)
  - [x] Update summary section
  - [x] Update file counts section
  - [x] Commit: "fix(plots): Separate exp5_3 plots by CAS latency" (a647bb7)

- [x] **exp3_2**: Investigate and fix distribution separation
  - [x] Check if exp3_2 experiments have been run (YES - all 36 experiments complete)
  - [x] Identify distribution type column in data (added conflicting_manifests_type)
  - [x] Apply appropriate filtering/grouping pattern
  - [x] Update regenerate_all_plots.sh
  - [x] Update summary and file counts sections
  - [x] Commit: "fix(plots): Separate exp3_2 plots by distribution type" (77012c9)

- [x] **Verification**: Create tool to validate filtering logic
  - [x] Created scripts/verify_filtered_analysis.py
  - [x] Validated 7 test cases across exp3_1, exp3_2, exp3_3, exp5_1, exp5_3
  - [x] All tests passed - filters correctly identify experiments and load data

- [ ] **Next**: Regenerate plots with corrected filtering
  - [ ] Run: ./scripts/regenerate_all_plots.sh --parallel 4
  - [ ] Verify exp3_2 generates 4 plot directories (one per distribution)
  - [ ] Verify exp5_3 generates 6 plot directories (one per CAS latency)
