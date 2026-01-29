#!/bin/bash
#
# Regenerate analysis plots for all experiment groups
#
# This script runs saturation analysis for all experiment patterns
# and generates plots with standard deviation across seeds.
#
# Uses consolidated.parquet file if available for efficient compressed storage.
# Falls back to individual results.parquet files if consolidated file missing.
#
# Usage:
#   ./scripts/regenerate_all_plots.sh [--parallel N]
#
# Options:
#   --parallel N    Number of parallel analysis jobs (default: 4)
#
# Note: To regenerate consolidated.parquet, run:
#   python scripts/consolidate_all_experiments_incremental.py
#

set -e

# Parse arguments
PARALLEL=4
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel N]"
            exit 1
            ;;
    esac
done

# Activate virtual environment
if [ -f bin/activate ]; then
    source bin/activate
else
    echo "Error: Virtual environment not found. Run from project root."
    exit 1
fi

echo "==================================="
echo "Regenerating Analysis Plots"
echo "==================================="
echo "Parallel jobs: $PARALLEL"
echo ""

# Create plots directory
mkdir -p plots

# Check if consolidated.parquet exists for faster processing
CONSOLIDATED_FILE="experiments/consolidated.parquet"
USE_CONSOLIDATED=false
TEMP_CONFIG=""

if [ -f "$CONSOLIDATED_FILE" ]; then
    echo "Found consolidated.parquet - using consolidated mode"
    USE_CONSOLIDATED=true

    # Create temporary config file to enable consolidated mode
    TEMP_CONFIG=$(mktemp)
    cat > "$TEMP_CONFIG" << 'CONFIGEOF'
[analysis]
use_consolidated = true

[paths]
consolidated_file = "experiments/consolidated.parquet"
CONFIGEOF

    trap "rm -f $TEMP_CONFIG" EXIT

    # Export for use in subshells
    export TEMP_CONFIG
    export USE_CONSOLIDATED
else
    echo "No consolidated.parquet found - using individual results files"
    echo "To reduce storage with consolidation, run: python scripts/consolidate_all_experiments_incremental.py"
fi
echo ""

# Define experiment configurations
# Format: "pattern|output_dir|group_by|description"
declare -a experiments=(
    "exp2_1_*|plots/exp2_1||Single-table false conflicts"
    "exp2_2_*|plots/exp2_2|num_tables|Multi-table false conflicts"
    # Note: exp3_1 handled separately below with filtering (sweeps real_conflict_probability)
    # Note: exp3_2 handled separately below with filtering (sweeps conflicting_manifests distribution)
    # Note: exp3_3 handled separately below with filtering (sweeps num_tables × real_conflict_probability)
    # Note: exp3_4 handled separately below with filtering (sweeps num_tables × real_conflict_probability)
    "exp4_1_*|plots/exp4_1||Single-table backoff comparison"
    "exp5_1_*|plots/exp5_1|t_cas_mean|Single-table catalog latency"
    # Note: exp5_2 handled separately below with filtering
    # Note: exp5_3 handled separately below with filtering (sweeps T_CAS × num_groups)
)

# Function to run analysis for one experiment group
run_analysis() {
    local pattern="$1"
    local output_dir="$2"
    local group_by="$3"
    local description="$4"

    # Check if any experiments exist matching the pattern
    local exp_count=$(find experiments -type d -name "$pattern" 2>/dev/null | wc -l)

    if [ "$exp_count" -eq 0 ]; then
        echo "Skipped: $description (no experiments found matching $pattern)"
        return 0
    fi

    echo "Starting: $description ($pattern) - $exp_count experiment configs found"

    # Create output directory
    mkdir -p "$output_dir"

    # Build command
    local cmd="python -m endive.saturation_analysis -i experiments -p \"$pattern\" -o \"$output_dir\""

    # Add config file if using consolidated mode
    if [ "$USE_CONSOLIDATED" = true ]; then
        cmd="$cmd -c \"$TEMP_CONFIG\""
    fi

    # Add group-by if specified
    if [ -n "$group_by" ]; then
        cmd="$cmd --group-by $group_by"
    fi

    # Run with logging
    eval "$cmd" > "$output_dir/analysis.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Completed: $description"
    else
        echo "Failed: $description (exit code: $exit_code)"
        # Don't propagate error - continue with other experiments
        return 0
    fi
}

export -f run_analysis

# Function to run analysis with filtering for one experiment group
run_analysis_with_filter() {
    local pattern="$1"
    local output_dir="$2"
    local group_by="$3"
    local filter_expr="$4"
    local description="$5"

    # Check if any experiments exist matching the pattern
    local exp_count=$(find experiments -type d -name "$pattern" 2>/dev/null | wc -l)

    if [ "$exp_count" -eq 0 ]; then
        echo "Skipped: $description (no experiments found matching $pattern)"
        return 0
    fi

    echo "Starting: $description ($pattern with filter '$filter_expr')"

    # Create output directory
    mkdir -p "$output_dir"

    # Build command
    local cmd="python -m endive.saturation_analysis -i experiments -p \"$pattern\" -o \"$output_dir\""

    # Add config file if using consolidated mode
    if [ "$USE_CONSOLIDATED" = true ]; then
        cmd="$cmd -c \"$TEMP_CONFIG\""
    fi

    # Add group-by if specified
    if [ -n "$group_by" ]; then
        cmd="$cmd --group-by $group_by"
    fi

    # Add filter
    cmd="$cmd --filter \"$filter_expr\""

    # Run with logging
    eval "$cmd" > "$output_dir/analysis.log" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Completed: $description"
    else
        echo "Failed: $description (exit code: $exit_code)"
        # Don't propagate error - continue with other experiments
        return 0
    fi
}

export -f run_analysis_with_filter

# Run analyses in background with limited parallelism
pids=()

# Handle exp3.1 with grouping by real_conflict_probability
# This generates 1 composite plot with all conflict probabilities
while [ ${#pids[@]} -ge "$PARALLEL" ]; do
    for i in "${!pids[@]}"; do
        if ! kill -0 "${pids[$i]}" 2>/dev/null; then
            unset 'pids[i]'
        fi
    done
    pids=("${pids[@]}")
    sleep 1
done

run_analysis \
    "exp3_1_*" \
    "plots/exp3_1" \
    "real_conflict_probability" \
    "Single-table real conflicts" &
pids+=($!)

# Handle exp3.2 with filtering by conflicting_manifests distribution type
# This generates 4 separate plots (one for each distribution configuration)
declare -a exp3_2_dist_types=("fixed-1" "fixed-5" "fixed-10" "exponential")
declare -a exp3_2_labels=("Fixed 1 manifest" "Fixed 5 manifests" "Fixed 10 manifests" "Exponential distribution")
for i in "${!exp3_2_dist_types[@]}"; do
    dist_type="${exp3_2_dist_types[$i]}"
    label="${exp3_2_labels[$i]}"

    # Wait if we've reached max parallel jobs
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        for j in "${!pids[@]}"; do
            if ! kill -0 "${pids[$j]}" 2>/dev/null; then
                unset 'pids[j]'
            fi
        done
        pids=("${pids[@]}")
        sleep 1
    done

    # Start analysis in background
    run_analysis_with_filter \
        "exp3_2_*" \
        "plots/exp3_2_${dist_type}" \
        "" \
        "conflicting_manifests_type=='${dist_type}'" \
        "Manifest distribution (${label})" &
    pids+=($!)
done

# Handle exp3.3 with grouping by real_conflict_probability and filtering by num_tables
# This generates 5 composite plots (one per table count, each with multiple conflict probabilities)
declare -a exp3_3_num_tables=(1 2 5 10 20)
for num_tables in "${exp3_3_num_tables[@]}"; do
    # Wait if we've reached max parallel jobs
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
        "exp3_3_*" \
        "plots/exp3_3_t${num_tables}" \
        "real_conflict_probability" \
        "num_tables==$num_tables" \
        "Multi-table real conflicts (${num_tables} tables)" &
    pids+=($!)
done

# Handle exp3.4 with grouping by real_conflict_probability and filtering by num_tables
# This generates 5 composite plots (one per table count, each with multiple conflict probabilities)
declare -a exp3_4_num_tables=(1 2 5 10 20)
for num_tables in "${exp3_4_num_tables[@]}"; do
    # Wait if we've reached max parallel jobs
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
        "exp3_4_*" \
        "plots/exp3_4_t${num_tables}" \
        "real_conflict_probability" \
        "num_tables==$num_tables" \
        "Exponential backoff with real conflicts (${num_tables} tables)" &
    pids+=($!)
done

# Handle exp5.2 with filtering by T_CAS value
# This generates 6 separate plots, one for each T_CAS value (15, 50, 100, 200, 500, 1000 ms)
declare -a exp5_2_t_cas_values=(15 50 100 200 500 1000)
for t_cas in "${exp5_2_t_cas_values[@]}"; do
    # Wait if we've reached max parallel jobs
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        # Check for completed jobs
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Job completed, remove from array
                unset 'pids[i]'
            fi
        done
        # Rebuild array to remove gaps
        pids=("${pids[@]}")
        sleep 1
    done

    # Start analysis in background
    run_analysis_with_filter \
        "exp5_2_*" \
        "plots/exp5_2_t_cas_${t_cas}ms" \
        "num_tables" \
        "t_cas_mean==$t_cas" \
        "Multi-table catalog latency (T_CAS=${t_cas}ms)" &
    pids+=($!)
done

# Handle exp5.3 with filtering by T_CAS value
# This generates 6 separate plots, one for each T_CAS value, with num_groups as series
declare -a exp5_3_t_cas_values=(15 50 100 200 500 1000)
for t_cas in "${exp5_3_t_cas_values[@]}"; do
    # Wait if we've reached max parallel jobs
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        # Check for completed jobs
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Job completed, remove from array
                unset 'pids[i]'
            fi
        done
        # Rebuild array to remove gaps
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
for exp_config in "${experiments[@]}"; do
    IFS='|' read -r pattern output_dir group_by description <<< "$exp_config"

    # Wait if we've reached max parallel jobs
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        # Check for completed jobs
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Job completed, remove from array
                unset 'pids[i]'
            fi
        done
        # Rebuild array to remove gaps
        pids=("${pids[@]}")
        sleep 1
    done

    # Start analysis in background
    run_analysis "$pattern" "$output_dir" "$group_by" "$description" &
    pids+=($!)
done

# Wait for all remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "==================================="
echo "Analysis Complete!"
echo "==================================="
echo ""

# Summary of generated files
echo "Generated files per experiment group:"
echo "  - experiment_index.csv (summary statistics)"
echo "  - latency_vs_throughput.png (with error bands)"
echo "  - latency_vs_throughput.md (with ± stddev)"
echo "  - success_vs_load.png"
echo "  - success_vs_throughput.png"
echo "  - overhead_vs_throughput.png (with error bands)"
echo "  - overhead_vs_throughput.md (with ± stddev)"
echo "  - commit_rate_over_time.png"
echo ""

# Show output locations
echo "Output locations:"
for exp_config in "${experiments[@]}"; do
    IFS='|' read -r pattern output_dir group_by description <<< "$exp_config"
    echo "  $description: $output_dir/"
done
# Add exp3.1 plot
echo "  Single-table real conflicts: plots/exp3_1/"
# Add exp3.2 plots
for i in "${!exp3_2_dist_types[@]}"; do
    dist_type="${exp3_2_dist_types[$i]}"
    label="${exp3_2_labels[$i]}"
    echo "  Manifest distribution (${label}): plots/exp3_2_${dist_type}/"
done
# Add exp3.3 plots
for num_tables in "${exp3_3_num_tables[@]}"; do
    echo "  Multi-table real conflicts (${num_tables} tables): plots/exp3_3_t${num_tables}/"
done
# Add exp3.4 plots
for num_tables in "${exp3_4_num_tables[@]}"; do
    echo "  Exponential backoff with real conflicts (${num_tables} tables): plots/exp3_4_t${num_tables}/"
done
# Add exp5.2 plots
for t_cas in "${exp5_2_t_cas_values[@]}"; do
    echo "  Multi-table catalog latency (T_CAS=${t_cas}ms): plots/exp5_2_t_cas_${t_cas}ms/"
done
# Add exp5.3 plots
for t_cas in "${exp5_3_t_cas_values[@]}"; do
    echo "  Transaction partitioning (T_CAS=${t_cas}ms): plots/exp5_3_t_cas_${t_cas}ms/"
done
echo ""

# Show file counts
echo "File summary:"
for exp_config in "${experiments[@]}"; do
    IFS='|' read -r pattern output_dir group_by description <<< "$exp_config"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
# Add exp3.1 file count
output_dir="plots/exp3_1"
if [ -d "$output_dir" ]; then
    png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
    md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
    csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
    echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
fi
# Add exp3.2 file counts
for dist_type in "${exp3_2_dist_types[@]}"; do
    output_dir="plots/exp3_2_${dist_type}"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
# Add exp3.3 file counts
for num_tables in "${exp3_3_num_tables[@]}"; do
    output_dir="plots/exp3_3_t${num_tables}"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
# Add exp3.4 file counts
for num_tables in "${exp3_4_num_tables[@]}"; do
    output_dir="plots/exp3_4_t${num_tables}"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
# Add exp5.2 file counts
for t_cas in "${exp5_2_t_cas_values[@]}"; do
    output_dir="plots/exp5_2_t_cas_${t_cas}ms"
    if [ -d "$output_dir" ]; then
        png_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
        md_count=$(find "$output_dir" -name "*.md" 2>/dev/null | wc -l)
        csv_count=$(find "$output_dir" -name "*.csv" 2>/dev/null | wc -l)
        echo "  $output_dir: $png_count PNGs, $md_count MDs, $csv_count CSVs"
    fi
done
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
echo ""

echo "View results:"
echo "  cat plots/exp2_1/latency_vs_throughput.md"
echo "  open plots/exp2_1/latency_vs_throughput.png"
echo ""
