#!/bin/bash
#
# Regenerate analysis plots for all experiment groups
#
# This script runs saturation analysis for all experiment patterns
# and generates plots with standard deviation across seeds.
#
# Usage:
#   ./scripts/regenerate_all_plots.sh [--parallel N]
#
# Options:
#   --parallel N    Number of parallel analysis jobs (default: 4)
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

# Define experiment configurations
# Format: "pattern|output_dir|group_by|description"
declare -a experiments=(
    "exp2_1_*|plots/exp2_1||Single-table false conflicts"
    "exp2_2_*|plots/exp2_2|num_tables|Multi-table false conflicts"
    "exp3_1_*|plots/exp3_1||Single-table real conflicts"
    "exp3_2_*|plots/exp3_2||Manifest distribution variance"
)

# Function to run analysis for one experiment group
run_analysis() {
    local pattern="$1"
    local output_dir="$2"
    local group_by="$3"
    local description="$4"

    echo "Starting: $description ($pattern)"

    # Create output directory
    mkdir -p "$output_dir"

    # Build command
    local cmd="python -m icecap.saturation_analysis -i experiments -p \"$pattern\" -o \"$output_dir\""

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
        return $exit_code
    fi
}

export -f run_analysis

# Run analyses in background with limited parallelism
pids=()
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
echo ""

echo "View results:"
echo "  cat plots/exp2_1/latency_vs_throughput.md"
echo "  open plots/exp2_1/latency_vs_throughput.png"
echo ""
