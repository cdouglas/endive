#!/bin/bash
#
# Re-run experiments with missing results.parquet files
#
# This script finds experiment directories with fewer than the expected number
# of seeds (default: 5) and re-runs the simulations for missing seeds.
#
# Usage:
#   ./scripts/rerun_missing_experiments.sh [--parallel N] [--expected-seeds N] [--dry-run]
#
# Options:
#   --parallel N         Number of parallel simulations (default: 4)
#   --expected-seeds N   Expected number of seeds per experiment (default: 5)
#   --dry-run           Show what would be done without running simulations
#

set -e

# Parse arguments
PARALLEL=4
EXPECTED_SEEDS=5
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --expected-seeds)
            EXPECTED_SEEDS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel N] [--expected-seeds N] [--dry-run]"
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

echo "========================================="
echo "Re-running Missing Experiments"
echo "========================================="
echo "Expected seeds per experiment: $EXPECTED_SEEDS"
echo "Parallel simulations: $PARALLEL"
echo "Dry run: $DRY_RUN"
echo ""

# Find all experiment directories with missing results
declare -a missing_experiments=()

echo "Scanning experiments directory..."
for exp_dir in experiments/exp*-[0-9a-f]*; do
    # Skip if not a directory
    if [ ! -d "$exp_dir" ]; then
        continue
    fi

    # Count results.parquet files
    result_count=$(find "$exp_dir" -name results.parquet 2>/dev/null | wc -l)

    if [ "$result_count" -lt "$EXPECTED_SEEDS" ]; then
        missing_count=$((EXPECTED_SEEDS - result_count))
        missing_experiments+=("$exp_dir:$missing_count")
        echo "  $(basename "$exp_dir"): $result_count/$EXPECTED_SEEDS results (missing $missing_count)"
    fi
done

if [ ${#missing_experiments[@]} -eq 0 ]; then
    echo ""
    echo "No missing experiments found. All experiments have $EXPECTED_SEEDS results."
    exit 0
fi

echo ""
echo "Found ${#missing_experiments[@]} experiments with missing results"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN - Would process the following:"
    echo ""
    for exp_info in "${missing_experiments[@]}"; do
        exp_dir="${exp_info%:*}"
        missing_count="${exp_info#*:}"
        echo "  $exp_dir (missing $missing_count)"

        # Find which seeds are missing
        for seed_dir in "$exp_dir"/[0-9]*; do
            if [ -d "$seed_dir" ]; then
                seed=$(basename "$seed_dir")
                if [ ! -f "$seed_dir/results.parquet" ]; then
                    echo "    → Would run seed $seed"
                fi
            fi
        done
    done
    echo ""
    echo "To run these experiments, execute without --dry-run flag:"
    echo "  ./scripts/rerun_missing_experiments.sh --parallel $PARALLEL"
    exit 0
fi

# Run missing simulations
echo "Starting simulations..."
echo ""

# Function to run a single simulation
run_simulation() {
    local exp_dir="$1"
    local seed="$2"
    local orig_dir="$(pwd)"
    local cfg_path="$(realpath "$exp_dir/cfg.toml")"
    local seed_dir="$(realpath "$exp_dir/$seed")"
    local log_file="$seed_dir/rerun.log"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $(basename "$exp_dir") seed=$seed"

    # Run simulation (cd to seed directory so output files go there)
    cd "$seed_dir"
    python -m icecap.main "$cfg_path" --seed "$seed" > "$log_file" 2>&1
    local exit_code=$?
    cd "$orig_dir"

    if [ $exit_code -eq 0 ] && [ -f "$seed_dir/results.parquet" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed: $(basename "$exp_dir") seed=$seed"
        return 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Failed: $(basename "$exp_dir") seed=$seed (exit code: $exit_code)"
        return 1
    fi
}

export -f run_simulation

# Build list of (exp_dir, seed) pairs to run
declare -a jobs=()
total_simulations=0

for exp_info in "${missing_experiments[@]}"; do
    exp_dir="${exp_info%:*}"

    # Check if cfg.toml exists
    if [ ! -f "$exp_dir/cfg.toml" ]; then
        echo "Warning: No cfg.toml in $exp_dir, skipping"
        continue
    fi

    # Find missing seeds
    for seed_dir in "$exp_dir"/[0-9]*; do
        if [ -d "$seed_dir" ]; then
            seed=$(basename "$seed_dir")
            if [ ! -f "$seed_dir/results.parquet" ]; then
                jobs+=("$exp_dir:$seed")
                total_simulations=$((total_simulations + 1))
            fi
        fi
    done
done

echo "Total simulations to run: $total_simulations"
echo "Running with parallelism: $PARALLEL"
echo ""

# Run simulations with limited parallelism
pids=()
completed=0
failed=0

for job in "${jobs[@]}"; do
    exp_dir="${job%:*}"
    seed="${job#*:}"

    # Wait if we've reached max parallel jobs
    while [ ${#pids[@]} -ge "$PARALLEL" ]; do
        # Check for completed jobs
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                # Job completed, check if it succeeded
                wait "${pids[$i]}"
                if [ $? -eq 0 ]; then
                    completed=$((completed + 1))
                else
                    failed=$((failed + 1))
                fi
                unset 'pids[i]'
            fi
        done
        # Rebuild array to remove gaps
        pids=("${pids[@]}")
        sleep 1
    done

    # Start simulation in background
    run_simulation "$exp_dir" "$seed" &
    pids+=($!)
done

# Wait for remaining jobs
for pid in "${pids[@]}"; do
    wait "$pid"
    if [ $? -eq 0 ]; then
        completed=$((completed + 1))
    else
        failed=$((failed + 1))
    fi
done

echo ""
echo "========================================="
echo "Re-run Complete"
echo "========================================="
echo "Total simulations: $total_simulations"
echo "Completed successfully: $completed"
echo "Failed: $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo "Some simulations failed. Check rerun.log files in seed directories for details."
    exit 1
fi

echo "All simulations completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Regenerate consolidated.parquet:"
echo "     python scripts/consolidate_all_experiments_incremental.py"
echo "  2. Regenerate plots:"
echo "     ./scripts/regenerate_all_plots.sh"
