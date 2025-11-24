#!/bin/bash
# run_baseline_experiments.sh
#
# Runs baseline experiments (Phase 2) from ANALYSIS_PLAN.md
#
# Usage:
#   ./run_baseline_experiments.sh [OPTIONS]
#
# Options:
#   --seeds N          Number of seeds per configuration (default: 3)
#   --parallel N       Number of concurrent experiments (default: # of CPU cores)
#   --exp2.1           Run only Experiment 2.1 (single table)
#   --exp2.2           Run only Experiment 2.2 (multi-table)
#   --quick            Quick test mode (fewer configs, shorter duration)
#   --dry-run          Show what would be run without executing
#
# Examples:
#   # Run all baseline experiments with 5 seeds each
#   ./run_baseline_experiments.sh --seeds 5
#
#   # Run with 8 parallel jobs
#   ./run_baseline_experiments.sh --parallel 8
#
#   # Run in background with logging
#   nohup ./run_baseline_experiments.sh --seeds 3 > experiments.log 2>&1 &
#
#   # Quick test run
#   ./run_baseline_experiments.sh --quick --seeds 1

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Default values
NUM_SEEDS=3
NUM_PARALLEL=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
RUN_EXP2_1=true
RUN_EXP2_2=true
QUICK_MODE=false
DRY_RUN=false

# Experiment parameters
EXP2_1_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_LOADS=(10 20 50 100 200 500 1000 2000 5000)
EXP2_2_TABLES=(1 2 5 10 20 50)

# Quick mode parameters (for testing)
QUICK_LOADS=(100 500 2000)
QUICK_TABLES=(1 5 10)
QUICK_DURATION=10000  # 10 seconds

# Base config files
EXP2_1_CONFIG="experiment_configs/exp2_1_single_table_false_conflicts.toml"
EXP2_2_CONFIG="experiment_configs/exp2_2_multi_table_false_conflicts.toml"

# Logging
LOG_DIR="experiment_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/baseline_experiments_${TIMESTAMP}.log"

# Job tracking
declare -a JOB_PIDS=()
declare -a JOB_CONFIGS=()
JOB_SUCCESS=0
JOB_FAILED=0

# ============================================================================
# Parse command line arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --parallel)
            NUM_PARALLEL="$2"
            shift 2
            ;;
        --exp2.1)
            RUN_EXP2_1=true
            RUN_EXP2_2=false
            shift
            ;;
        --exp2.2)
            RUN_EXP2_1=false
            RUN_EXP2_2=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -n 25 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Setup
# ============================================================================

# Create log directory
mkdir -p "$LOG_DIR"

# Activate virtual environment
if [ ! -f "bin/activate" ]; then
    echo "Error: Virtual environment not found. Please run from project root."
    exit 1
fi

source bin/activate

# Verify icecap module is available
if ! python -c "import icecap.main" 2>/dev/null; then
    echo "Error: icecap module not found. Please install with 'pip install -e .'"
    exit 1
fi

# ============================================================================
# Helper Functions
# ============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_section() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "$*" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
}

estimate_duration() {
    local num_runs=$1
    local avg_time_per_run=$2  # in seconds
    local num_parallel=$3

    # Calculate wall-clock time with parallelism
    local total_seconds=$(( (num_runs * avg_time_per_run + num_parallel - 1) / num_parallel ))

    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))

    if [ $hours -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    else
        echo "${minutes}m"
    fi
}

run_experiment_background() {
    local config_file=$1
    local output_file=$2
    local description=$3

    if [ "$DRY_RUN" = true ]; then
        log "[DRY RUN] Would run: python -m icecap.main $output_file"
        return 0
    fi

    # Run simulation in background (auto-confirm with 'Y', disable progress bar for clean logs)
    {
        echo "Y" | python -m icecap.main --no-progress "$output_file" >> "$LOG_FILE" 2>&1
        exit_code=$?

        # Write result to temp file
        echo "$exit_code|$description" > "$output_file.status"
        exit $exit_code
    } &

    local pid=$!
    JOB_PIDS+=($pid)
    JOB_CONFIGS+=("$description")
}

wait_for_job_slot() {
    # Wait until we have fewer than NUM_PARALLEL running jobs
    while [ ${#JOB_PIDS[@]} -ge $NUM_PARALLEL ]; do
        check_completed_jobs
        sleep 1
    done
}

check_completed_jobs() {
    local new_pids=()
    local new_configs=()

    for i in "${!JOB_PIDS[@]}"; do
        local pid=${JOB_PIDS[$i]}
        local config="${JOB_CONFIGS[$i]}"

        if ! kill -0 $pid 2>/dev/null; then
            # Job has finished, check status
            wait $pid
            local exit_code=$?

            if [ $exit_code -eq 0 ]; then
                log "  ✓ Success: $config"
                JOB_SUCCESS=$((JOB_SUCCESS + 1))
            else
                log "  ✗ Failed: $config"
                JOB_FAILED=$((JOB_FAILED + 1))
            fi
        else
            # Job still running, keep it in the list
            new_pids+=($pid)
            new_configs+=("$config")
        fi
    done

    JOB_PIDS=("${new_pids[@]}")
    JOB_CONFIGS=("${new_configs[@]}")
}

wait_for_all_jobs() {
    log ""
    log "Waiting for all jobs to complete..."

    while [ ${#JOB_PIDS[@]} -gt 0 ]; do
        check_completed_jobs
        if [ ${#JOB_PIDS[@]} -gt 0 ]; then
            log "  ${#JOB_PIDS[@]} jobs still running..."
            sleep 5
        fi
    done

    log "All jobs complete"
}

create_config_variant() {
    local base_config=$1
    local output_file=$2
    shift 2
    local modifications=("$@")

    # Copy base config
    cp "$base_config" "$output_file"

    # Apply modifications
    for mod in "${modifications[@]}"; do
        local param=$(echo "$mod" | cut -d'=' -f1)
        local value=$(echo "$mod" | cut -d'=' -f2)

        # Use sed to replace parameter value
        # Handle both "param = value" and "param=value" formats
        sed -i "s/^${param}[[:space:]]*=.*/${param} = ${value}/" "$output_file"
    done
}

# ============================================================================
# Calculate total runs
# ============================================================================

TOTAL_RUNS=0

if [ "$QUICK_MODE" = true ]; then
    log "Running in QUICK MODE (reduced parameters for testing)"
    EXP2_1_LOADS=("${QUICK_LOADS[@]}")
    EXP2_2_LOADS=("${QUICK_LOADS[@]}")
    EXP2_2_TABLES=("${QUICK_TABLES[@]}")
fi

if [ "$RUN_EXP2_1" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_1_LOADS[@]} * NUM_SEEDS))
fi

if [ "$RUN_EXP2_2" = true ]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXP2_2_LOADS[@]} * ${#EXP2_2_TABLES[@]} * NUM_SEEDS))
fi

# ============================================================================
# Print Summary
# ============================================================================

log_section "BASELINE EXPERIMENTS - PHASE 2 (PARALLEL EXECUTION)"
log "Configuration:"
log "  Number of seeds per config: $NUM_SEEDS"
log "  Parallel jobs: $NUM_PARALLEL"
log "  Run Experiment 2.1: $RUN_EXP2_1"
log "  Run Experiment 2.2: $RUN_EXP2_2"
log "  Quick mode: $QUICK_MODE"
log "  Dry run: $DRY_RUN"
log ""
log "Total simulations to run: $TOTAL_RUNS"

if [ "$QUICK_MODE" = true ]; then
    AVG_TIME=15  # 15 seconds per quick run
else
    AVG_TIME=3600  # 1 hour per full run (updated for new duration)
fi

ESTIMATED_TIME=$(estimate_duration $TOTAL_RUNS $AVG_TIME $NUM_PARALLEL)
log "Estimated wall-clock time: $ESTIMATED_TIME (with $NUM_PARALLEL parallel jobs)"
log ""
log "Log file: $LOG_FILE"
log "Output directory: experiments/"

if [ "$DRY_RUN" = false ]; then
    log ""
    log "Starting experiments in 3 seconds... (Ctrl+C to cancel)"
    sleep 3
fi

# ============================================================================
# Experiment 2.1: Single Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_1" = true ]; then
    log_section "EXPERIMENT 2.1: Single Table Saturation (False Conflicts)"
    log "Research Question: What is the maximum throughput for a single table?"
    log ""
    log "Sweeping inter_arrival.scale: ${EXP2_1_LOADS[*]}"
    log "Seeds per configuration: $NUM_SEEDS"
    log "Running up to $NUM_PARALLEL experiments in parallel"
    log ""

    CURRENT_RUN=0

    for load in "${EXP2_1_LOADS[@]}"; do
        log "Load: inter_arrival.scale = ${load}ms (~$((1000/load)) txn/sec offered)"

        for seed_num in $(seq 1 $NUM_SEEDS); do
            CURRENT_RUN=$((CURRENT_RUN + 1))
            PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

            # Create temporary config with modified parameters
            TEMP_CONFIG=$(mktemp)

            if [ "$QUICK_MODE" = true ]; then
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "inter_arrival.scale=${load}" \
                    "duration_ms=${QUICK_DURATION}"
            else
                create_config_variant "$EXP2_1_CONFIG" "$TEMP_CONFIG" \
                    "inter_arrival.scale=${load}"
            fi

            # Wait for available job slot
            wait_for_job_slot

            # Run experiment in background
            DESC="$PROGRESS Exp2.1 load=$load seed=$seed_num"
            log "  Starting: $DESC"
            run_experiment_background "$EXP2_1_CONFIG" "$TEMP_CONFIG" "$DESC"
        done
    done

    log ""
    log "All Experiment 2.1 jobs submitted"
fi

# ============================================================================
# Experiment 2.2: Multi-Table Saturation (False Conflicts)
# ============================================================================

if [ "$RUN_EXP2_2" = true ]; then
    log_section "EXPERIMENT 2.2: Multi-Table Saturation (False Conflicts)"
    log "Research Question: How does table count affect throughput?"
    log ""
    log "Sweeping num_tables: ${EXP2_2_TABLES[*]}"
    log "Sweeping inter_arrival.scale: ${EXP2_2_LOADS[*]}"
    log "Seeds per configuration: $NUM_SEEDS"
    log "Running up to $NUM_PARALLEL experiments in parallel"
    log ""

    for num_tables in "${EXP2_2_TABLES[@]}"; do
        log "Tables: num_tables = $num_tables (num_groups = $num_tables)"

        for load in "${EXP2_2_LOADS[@]}"; do
            log "  Load: inter_arrival.scale = ${load}ms"

            for seed_num in $(seq 1 $NUM_SEEDS); do
                CURRENT_RUN=$((CURRENT_RUN + 1))
                PROGRESS="[$CURRENT_RUN/$TOTAL_RUNS]"

                # Create temporary config with modified parameters
                TEMP_CONFIG=$(mktemp)

                if [ "$QUICK_MODE" = true ]; then
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}" \
                        "duration_ms=${QUICK_DURATION}"
                else
                    create_config_variant "$EXP2_2_CONFIG" "$TEMP_CONFIG" \
                        "num_tables=${num_tables}" \
                        "num_groups=${num_tables}" \
                        "inter_arrival.scale=${load}"
                fi

                # Wait for available job slot
                wait_for_job_slot

                # Run experiment in background
                DESC="$PROGRESS Exp2.2 tables=$num_tables load=$load seed=$seed_num"
                log "  Starting: $DESC"
                run_experiment_background "$EXP2_2_CONFIG" "$TEMP_CONFIG" "$DESC"
            done
        done
    done

    log ""
    log "All Experiment 2.2 jobs submitted"
fi

# ============================================================================
# Wait for all jobs to complete
# ============================================================================

wait_for_all_jobs

# ============================================================================
# Final Summary
# ============================================================================

log_section "BASELINE EXPERIMENTS COMPLETE"
log "Total runs: $TOTAL_RUNS"
log "  Successful: $JOB_SUCCESS"
log "  Failed: $JOB_FAILED"
log ""
log "Results stored in: experiments/"
log "  exp2_1_single_table_false-*/"
log "  exp2_2_multi_table_false-*/"
log ""
log "Note: Interrupted runs leave .running.parquet files that can be safely deleted"
log ""
log "Next steps:"
log "  1. Verify results: find experiments/ -name 'results.parquet' | wc -l"
log "  2. Check for incomplete runs: find experiments/ -name '.running.parquet'"
log "  3. Run analysis: python -m icecap.saturation_analysis -i experiments -p 'exp2_1_*' -o plots/exp2_1"
log "  4. Generate visualizations for all experiments"

if [ "$DRY_RUN" = true ]; then
    log ""
    log "DRY RUN COMPLETE - No experiments were actually executed"
fi
