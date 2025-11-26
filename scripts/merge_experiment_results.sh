#!/bin/bash
#
# Merge experiment results from remote execution back into local experiments/
#
# This script takes results from a remote execution batch and merges them
# into the local experiments directory structure, then updates consolidated.parquet.
#
# Usage:
#   ./scripts/merge_experiment_results.sh RESULTS_DIR [--dry-run]
#
# Options:
#   --dry-run    Show what would be done without actually copying files
#

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 RESULTS_DIR [--dry-run]"
    echo ""
    echo "Example:"
    echo "  ./scripts/merge_experiment_results.sh results/"
    exit 1
fi

RESULTS_DIR="$1"
DRY_RUN=false

if [ $# -gt 1 ] && [ "$2" = "--dry-run" ]; then
    DRY_RUN=true
fi

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "========================================="
echo "Merging Experiment Results"
echo "========================================="
echo "Results directory: $RESULTS_DIR"
echo "Target directory: experiments/"
echo "Dry run: $DRY_RUN"
echo ""

# Scan results directory
echo "Scanning results..."

merged_count=0
skipped_count=0
error_count=0

for exp_result_dir in "$RESULTS_DIR"/exp*-[0-9a-f]*; do
    if [ ! -d "$exp_result_dir" ]; then
        continue
    fi

    exp_name=$(basename "$exp_result_dir")
    target_exp_dir="experiments/$exp_name"

    if [ ! -d "$target_exp_dir" ]; then
        echo "Warning: Target experiment directory not found: $target_exp_dir"
        error_count=$((error_count + 1))
        continue
    fi

    # Process each seed
    for seed_result_dir in "$exp_result_dir"/[0-9]*; do
        if [ ! -d "$seed_result_dir" ]; then
            continue
        fi

        seed=$(basename "$seed_result_dir")
        results_file="$seed_result_dir/results.parquet"
        target_seed_dir="$target_exp_dir/$seed"

        if [ ! -f "$results_file" ]; then
            echo "  ⚠ Skipped: $exp_name/$seed (no results.parquet)"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if [ ! -d "$target_seed_dir" ]; then
            echo "  ⚠ Skipped: $exp_name/$seed (target seed directory not found)"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        if [ -f "$target_seed_dir/results.parquet" ]; then
            echo "  ⚠ Skipped: $exp_name/$seed (results already exist)"
            skipped_count=$((skipped_count + 1))
            continue
        fi

        # Copy results
        if [ "$DRY_RUN" = true ]; then
            echo "  → Would copy: $exp_name/$seed"
        else
            cp "$results_file" "$target_seed_dir/results.parquet"
            echo "  ✓ Merged: $exp_name/$seed"
        fi
        merged_count=$((merged_count + 1))
    done
done

echo ""
echo "========================================="
echo "Merge Summary"
echo "========================================="
echo "Files merged: $merged_count"
echo "Files skipped: $skipped_count"
echo "Errors: $error_count"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - No files were actually copied."
    echo "Run without --dry-run to perform the merge."
    exit 0
fi

if [ $merged_count -eq 0 ]; then
    echo "No files were merged."
    exit 0
fi

# Regenerate consolidated.parquet
echo "Regenerating consolidated.parquet..."
echo ""

if [ -f bin/activate ]; then
    source bin/activate

    if [ -f scripts/consolidate_all_experiments_incremental.py ]; then
        python scripts/consolidate_all_experiments_incremental.py
        echo ""
        echo "✓ Consolidated.parquet updated"
    else
        echo "Warning: Consolidation script not found. Skipping."
    fi
else
    echo "Warning: Virtual environment not found. Skipping consolidation."
    echo "Run manually: python scripts/consolidate_all_experiments_incremental.py"
fi

echo ""
echo "========================================="
echo "Merge Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Verify merged results:"
echo "     find experiments -name results.parquet | wc -l"
echo "  2. Regenerate plots (if needed):"
echo "     ./scripts/regenerate_all_plots.sh"
